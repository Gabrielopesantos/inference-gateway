//! Runs every SQL query of the provisioned dashboards on a seeded store, as
//! Grafana would with all variable values selected.
#![cfg(feature = "store")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use inference_gateway::store::open_writer;
use rusqlite::types::Value;
use rusqlite::{Connection, params};
use serde_json::Value as Json;

const SQLITE_UID: &str = "gateway-sqlite";

fn seed(conn: &Connection) {
    let trace = |id: u8, model: &str, ttft_ms: Option<f64>, outcome: &str| {
        conn.execute(
            "INSERT INTO traces (id, started_at, upstream, provider, method, path, response_model,
                                 streaming, session_id, tags, outcome, stop_reason,
                                 input_tokens_total, output_tokens, cache_read_tokens, ttft_ms,
                                 duration_ms, chunk_count, output_tps, cost_usd, captured)
             VALUES (?1, ?2, 'a', 'anthropic', 'POST', '/v1/messages', ?3, 1, 's1', '{}', ?4,
                     'end_turn', 100, 10, 50, ?5, 1000, 3, 20, 0.01, 1)",
            params![[id], 1_000_000 + i64::from(id), model, outcome, ttft_ms],
        )
        .unwrap();
    };
    // m-a: 100..2000 ms, two failures; m-b: 3000..6000 ms and one without a first token.
    for i in 1..=20u8 {
        let outcome = if i <= 2 { "upstream_error" } else { "ok" };
        trace(i, "m-a", Some(f64::from(i) * 100.0), outcome);
    }
    for i in 0..4u8 {
        trace(21 + i, "m-b", Some(f64::from(i + 3) * 1000.0), "ok");
    }
    trace(25, "m-b", None, "ok");

    conn.execute_batch(
        r#"INSERT INTO evaluators (name, kind, pass, min_confidence, updated_at) VALUES
             ('refused', 'yes_no', '["no"]', NULL, 0),
             ('answered', 'choice', '["correct","partial"]', 0.7, 0);
           INSERT INTO eval_jobs (trace_id, evaluator, evaluator_version, status, attempts,
                                  not_before, last_error, created_at)
             VALUES (x'15', 'answered', 'v1', 'failed', 5, 0, 'timeout', 1000000);"#,
    )
    .unwrap();
    let scores = [
        ("refused", "bool", Some(false), None, Some(0.2), None),
        ("refused", "bool", Some(true), None, Some(0.8), None),
        (
            "answered",
            "categorical",
            None,
            Some("partial"),
            None,
            Some(0.9),
        ),
        (
            "answered",
            "categorical",
            None,
            Some("wrong"),
            None,
            Some(0.9),
        ),
        (
            "answered",
            "categorical",
            None,
            Some("correct"),
            None,
            Some(0.5),
        ),
        ("level", "numeric", None, Some("good"), Some(1.2), Some(0.9)),
    ];
    for (i, (name, data_type, b, label, num, confidence)) in scores.iter().enumerate() {
        conn.execute(
            "INSERT INTO scores (trace_id, name, data_type, value_bool, value_label, value_num,
                                 confidence, distribution, source, evaluator_version,
                                 judge_model, judge_latency_ms, judge_input_tokens,
                                 judge_cost_usd, input_truncated, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, '{}', 'judge', 'v1', 'jev', 100, 500, 0.001, 0,
                     1000000)",
            params![
                [u8::try_from(i + 3).unwrap()],
                name,
                data_type,
                b,
                label,
                num,
                confidence
            ],
        )
        .unwrap();
    }
}

fn rows(conn: &Connection, sql: &str) -> Vec<Vec<Value>> {
    let mut stmt = conn.prepare(sql).unwrap_or_else(|e| panic!("{e}\n{sql}"));
    let n = stmt.column_count();
    stmt.query_map([], |r| (0..n).map(|i| r.get(i)).collect())
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap_or_else(|e| panic!("{e}\n{sql}"))
}

fn is_sqlite(v: &Json) -> bool {
    v["datasource"]["uid"] == SQLITE_UID
}

/// Fills the time range and variables the way Grafana does with "All".
struct Vars(BTreeMap<String, String>);

impl Vars {
    fn new(conn: &Connection, dashboard: &Json) -> Self {
        let mut vars = Self(BTreeMap::new());
        for v in dashboard["templating"]["list"].as_array().unwrap() {
            assert!(is_sqlite(v), "variable {} must query the store", v["name"]);
            let name = v["name"].as_str().unwrap();
            let sql = vars.fill(v["query"].as_str().unwrap());
            let values: Vec<String> = rows(conn, &sql)
                .into_iter()
                .map(|r| match &r[0] {
                    Value::Text(s) => s.clone(),
                    other => panic!("{name}: {other:?}"),
                })
                .collect();
            assert!(!values.is_empty(), "variable {name} has no values");
            let csv = v["allValue"]
                .as_str()
                .map_or_else(|| values.join(","), str::to_owned);
            let quoted = values
                .iter()
                .map(|s| format!("'{s}'"))
                .collect::<Vec<_>>()
                .join(",");
            vars.0.insert(format!("${{{name}:csv}}"), csv);
            vars.0.insert(format!("${{{name}:singlequote}}"), quoted);
        }
        vars
    }

    fn fill(&self, sql: &str) -> String {
        let mut out = sql
            .replace("$__interval_ms", "60000")
            .replace("$__from", "0")
            .replace("$__to", "9999999999999");
        for (k, v) in &self.0 {
            out = out.replace(k, v);
        }
        assert!(!out.contains('$'), "unfilled variable:\n{out}");
        out
    }
}

/// Every SQL panel's rows by title.
fn run(conn: &Connection, file: &str) -> BTreeMap<String, Vec<Vec<Value>>> {
    let path = format!(
        "{}/deploy/grafana/dashboards/{file}",
        env!("CARGO_MANIFEST_DIR")
    );
    let dashboard: Json = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let vars = Vars::new(conn, &dashboard);
    let mut out = BTreeMap::new();
    for panel in dashboard["panels"].as_array().unwrap() {
        let title = panel["title"].as_str().unwrap().to_owned();
        for t in panel["targets"].as_array().into_iter().flatten() {
            if !is_sqlite(t) {
                continue;
            }
            assert_eq!(t["rawQueryText"], t["queryText"], "{title}");
            let sql = vars.fill(t["rawQueryText"].as_str().unwrap());
            assert!(
                out.insert(title.clone(), rows(conn, &sql)).is_none(),
                "duplicate title {title}"
            );
        }
    }
    out
}

fn real(v: &Value) -> f64 {
    match v {
        Value::Real(f) => *f,
        #[allow(clippy::cast_precision_loss)]
        Value::Integer(i) => *i as f64,
        other => panic!("not a number: {other:?}"),
    }
}

fn single(panels: &BTreeMap<String, Vec<Vec<Value>>>, title: &str) -> f64 {
    let rows = &panels[title];
    assert_eq!(rows.len(), 1, "{title}: {rows:?}");
    real(&rows[0][0])
}

fn db() -> (tempfile::TempDir, Connection) {
    let dir = tempfile::tempdir().unwrap();
    let conn = open_writer(&dir.path().join("g.db")).unwrap();
    seed(&conn);
    (dir, conn)
}

#[test]
fn gateway_sql_panels() {
    let (_dir, conn) = db();
    let p = run(&conn, "gateway.json");

    let requests: f64 = p["Requests"].iter().map(|r| real(&r[1])).sum();
    assert!((requests - 25.0).abs() < f64::EPSILON);
    assert!((single(&p, "Error ratio") - 0.08).abs() < 1e-9);
    assert!((single(&p, "Cache hit ratio") - 0.5).abs() < 1e-9);
    // Nearest rank: ceil(0.95 * 24) = 23rd of 100..2000, 3000..6000.
    assert!((single(&p, "TTFT p95") - 5000.0).abs() < f64::EPSILON);
    assert!((single(&p, "Sessions") - 1.0).abs() < f64::EPSILON);

    // model, requests, error_ratio, ttft_p50_ms, ttft_p95_ms, ...
    let by_model: BTreeMap<String, (f64, f64, f64)> = p["By model"]
        .iter()
        .map(|r| {
            let Value::Text(m) = &r[0] else { panic!() };
            (m.clone(), (real(&r[1]), real(&r[3]), real(&r[4])))
        })
        .collect();
    assert_eq!(by_model["m-a"], (20.0, 1000.0, 1900.0));
    assert_eq!(by_model["m-b"], (5.0, 4000.0, 6000.0));
}

#[test]
fn quality_sql_panels() {
    let (_dir, conn) = db();
    let p = run(&conn, "quality.json");

    assert!((single(&p, "Scores") - 6.0).abs() < f64::EPSILON);
    // pass: refused false, partial; fail: refused true, wrong; uncertain: correct.
    assert!((single(&p, "Pass rate") - 0.5).abs() < 1e-9);
    assert!((single(&p, "Uncertain") - 0.2).abs() < 1e-9);
    // time, model, evaluator, verdict, ...
    let review: Vec<_> = p["Fail or uncertain"]
        .iter()
        .map(|r| (r[2].clone(), r[3].clone()))
        .collect();
    assert_eq!(review.len(), 3, "{review:?}");
    assert_eq!(p["Failed jobs"].len(), 1);
}
