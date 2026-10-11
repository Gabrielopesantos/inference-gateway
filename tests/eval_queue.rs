#![cfg(feature = "eval")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::time::Duration;

use axum::http::StatusCode;
use inference_gateway::eval::queue::{self, Job, Score};
use inference_gateway::store::open_writer;
use rusqlite::{Connection, params};

use support::mock_upstream::{MockUpstream, Script, unreachable_addr};
use support::{Options, client, config, spawn_gateway_with};

const LEASE: Duration = Duration::from_secs(1);
const V: &str = "v1";

fn db() -> (tempfile::TempDir, Connection) {
    let dir = tempfile::tempdir().unwrap();
    let conn = open_writer(&dir.path().join("g.db")).unwrap();
    (dir, conn)
}

fn job(conn: &Connection, trace: u8) {
    conn.execute(
        "INSERT INTO traces (id, started_at, upstream, provider, method, path, streaming,
                             tags, outcome, duration_ms, chunk_count, captured)
         VALUES (?1, 0, 'a', 'anthropic', 'POST', '/v1/messages', 1, '{}', 'ok', 1, 1, 1)",
        [[trace]],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO eval_jobs (trace_id, evaluator, evaluator_version, status, not_before, created_at)
         VALUES (?1, 'e', ?2, 'pending', 0, 0)",
        params![[trace], V],
    )
    .unwrap();
}

fn score() -> Score {
    Score {
        data_type: "categorical",
        value_bool: None,
        value_label: Some("ok".into()),
        value_num: None,
        confidence: Some(0.9),
        distribution: "{}".into(),
        judge_model: "jev-1.13.0".into(),
        judge_latency_ms: 10.0,
        judge_input_tokens: 100,
        judge_cost_usd: 0.0,
        input_truncated: false,
    }
}

fn status(conn: &Connection) -> (String, i64) {
    conn.query_row("SELECT status, attempts FROM eval_jobs", [], |r| {
        Ok((r.get(0)?, r.get(1)?))
    })
    .unwrap()
}

fn claim(conn: &Connection, now: i64) -> Vec<Job> {
    queue::claim(conn, &[V.to_owned()], 10, LEASE, now).unwrap()
}

#[test]
fn expired_lease_is_reclaimed_and_scored_once() {
    let (_dir, mut conn) = db();
    job(&conn, 1);
    let first = claim(&conn, 0);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].attempts, 1);
    assert!(claim(&conn, 500).is_empty(), "lease still held");

    let second = claim(&conn, 2000);
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].attempts, 2);

    // The first worker comes back after losing its lease.
    assert!(!queue::complete(&mut conn, &first[0], &score(), 2100).unwrap());
    assert!(queue::complete(&mut conn, &second[0], &score(), 2200).unwrap());
    assert!(!queue::complete(&mut conn, &second[0], &score(), 2300).unwrap());
    let scores: i64 = conn
        .query_row("SELECT count(*) FROM scores", [], |r| r.get(0))
        .unwrap();
    assert_eq!(scores, 1);
    assert_eq!(status(&conn), ("done".into(), 2));
}

#[test]
fn retry_waits_then_fails_at_max_attempts() {
    let (_dir, mut conn) = db();
    job(&conn, 1);
    let j = claim(&conn, 0).remove(0);
    assert!(!queue::retry(&mut conn, &j, 2, Duration::from_secs(3), "429", 0).unwrap());
    assert_eq!(status(&conn), ("pending".into(), 1));
    assert!(claim(&conn, 2999).is_empty(), "not before 3 s");
    let j = claim(&conn, 3000).remove(0);
    assert!(queue::retry(&mut conn, &j, 2, Duration::ZERO, "429 again", 3000).unwrap());
    assert_eq!(status(&conn), ("failed".into(), 2));
    assert!(claim(&conn, 10_000).is_empty());
    let err: String = conn
        .query_row("SELECT last_error FROM eval_jobs", [], |r| r.get(0))
        .unwrap();
    assert_eq!(err, "429 again");
}

#[test]
fn release_returns_the_attempt() {
    let (_dir, mut conn) = db();
    job(&conn, 1);
    let j = claim(&conn, 0).remove(0);
    queue::release(&mut conn, &j, 100).unwrap();
    assert_eq!(status(&conn), ("pending".into(), 0));
    assert_eq!(claim(&conn, 100)[0].attempts, 1);
}

#[test]
fn other_versions_are_not_claimed() {
    let (_dir, conn) = db();
    job(&conn, 1);
    assert!(
        queue::claim(&conn, &["v2".to_owned()], 10, LEASE, 0)
            .unwrap()
            .is_empty()
    );
    assert_eq!(claim(&conn, 0).len(), 1);
}

#[test]
fn verdicts_follow_the_registered_pass_rule() {
    let (_dir, mut conn) = db();
    let cfg = |pass: &str| {
        inference_gateway::config::Config::parse(&config(
            &[("a", "anthropic", "http://x")],
            &format!(
                r#"[store]
path = "g.db"

[eval.judge]
base_url = "https://x"
model = "jev"
api_key_env = "K"

[[eval.evaluators]]
name = "refused"
state = "x"
pass = "no"
question = {{ type = "noul", instructions = "Refused?" }}

[[eval.evaluators]]
name = "answered"
state = "x"
min_confidence = 0.7
{pass}
question = {{ type = "choice", instructions = "Answered?", criteria = {{ correct = "", partial = "", wrong = "" }} }}

[[eval.evaluators]]
name = "level"
state = "x"
pass = ["good"]
question = {{ type = "score", instructions = "Level?", criteria = ["bad", "good"] }}
"#
            ),
        ))
        .unwrap()
        .eval
        .unwrap()
        .evaluators
    };
    queue::register(&mut conn, &cfg("pass = [\"correct\", \"partial\"]"), 1).unwrap();

    let rows = [
        ("refused", "bool", Some(false), None, None),
        ("refused", "bool", Some(true), None, None),
        ("answered", "categorical", None, Some("partial"), Some(0.9)),
        ("answered", "categorical", None, Some("wrong"), Some(0.9)),
        ("answered", "categorical", None, Some("correct"), Some(0.5)),
        ("level", "numeric", None, Some("good"), Some(0.9)),
        ("unknown", "categorical", None, Some("x"), Some(0.9)),
    ];
    for (i, (name, data_type, b, label, confidence)) in rows.iter().enumerate() {
        conn.execute(
            "INSERT INTO traces (id, started_at, upstream, provider, method, path, streaming,
                                 tags, outcome, duration_ms, chunk_count, captured)
             VALUES (?1, 0, 'a', 'anthropic', 'POST', '/v1/messages', 1, '{}', 'ok', 1, 1, 1)",
            [[u8::try_from(i).unwrap()]],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO scores (trace_id, name, data_type, value_bool, value_label, confidence,
                                 source, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'judge', 0)",
            params![
                [u8::try_from(i).unwrap()],
                name,
                data_type,
                b,
                label,
                confidence
            ],
        )
        .unwrap();
    }
    let verdicts = |conn: &Connection| -> Vec<(String, Option<String>)> {
        conn.prepare("SELECT label, verdict FROM score_verdicts ORDER BY id")
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap()
    };
    let expect = |v: &[(&str, Option<&str>)]| -> Vec<(String, Option<String>)> {
        v.iter()
            .map(|(l, v)| ((*l).to_owned(), v.map(str::to_owned)))
            .collect()
    };
    assert_eq!(
        verdicts(&conn),
        expect(&[
            ("false", Some("pass")),
            ("true", Some("fail")),
            ("partial", Some("pass")),
            ("wrong", Some("fail")),
            ("correct", Some("uncertain")),
            ("good", Some("pass")),
            ("x", None),
        ])
    );

    // A changed rule applies to stored scores; no rule gives no verdict.
    queue::register(&mut conn, &cfg(""), 2).unwrap();
    assert_eq!(verdicts(&conn)[2], ("partial".to_owned(), None));
}

#[test]
fn claim_takes_at_most_n_in_id_order() {
    let (_dir, conn) = db();
    for t in 1..=3 {
        job(&conn, t);
    }
    let jobs = queue::claim(&conn, &[V.to_owned()], 2, LEASE, 0).unwrap();
    assert_eq!(
        jobs.iter().map(|j| j.trace_id.clone()).collect::<Vec<_>>(),
        [vec![1], vec![2]]
    );
}

const TEXT_SSE: &[u8] = include_bytes!("fixtures/anthropic/text.sse");

/// Which requests get a job: captured, outcome ok, `match` and `sample`.
#[tokio::test]
async fn enqueue_follows_capture_outcome_match_and_sample() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("g.db");
    let ok = MockUpstream::start(Script::sse(TEXT_SSE, Duration::ZERO)).await;
    let failing = MockUpstream::start(Script::json(StatusCode::INTERNAL_SERVER_ERROR, b"{}")).await;
    let extra = format!(
        r#"
[store]
path = "{}"

[capture]

[[capture.rules]]
match = {{ tags = {{ capture = "no" }} }}
capture = false

[[capture.rules]]

[eval]
poll = "1h"

[eval.judge]
base_url = "http://{}"
model = "jev-1.13.0"
api_key_env = "IG_TEST_UNUSED"

[[eval.evaluators]]
name = "tagged"
match = {{ tags = {{ project = "ig" }} }}
state = "{{{{ output_text }}}}"
question = {{ type = "noul", instructions = "?" }}

[[eval.evaluators]]
name = "never"
sample = 0.0
state = "x"
question = {{ type = "noul", instructions = "?" }}
"#,
        db.display(),
        unreachable_addr()
    );
    let gw = spawn_gateway_with(
        &config(
            &[
                ("ok", "anthropic", &ok.base_url()),
                ("failing", "anthropic", &failing.base_url()),
            ],
            &extra,
        ),
        Options {
            judge_key: Some("k".into()),
            ..Options::default()
        },
    )
    .await;
    let body =
        r#"{"model":"claude-sonnet-5","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    for (upstream, tags) in [
        ("ok", &[("project", "ig")][..]),                    // job
        ("ok", &[("project", "other")][..]),                 // no match
        ("ok", &[("project", "ig"), ("capture", "no")][..]), // not captured
        ("failing", &[("project", "ig")][..]),               // outcome
    ] {
        let mut req = client()
            .post(gw.url(&format!("/{upstream}/v1/messages")))
            .body(body);
        for (k, v) in tags {
            req = req.header(format!("x-gateway-tag-{k}"), *v);
        }
        req.send().await.unwrap().bytes().await.unwrap();
    }
    gw.shutdown(Duration::from_secs(5)).await.unwrap();

    let conn = open_writer(&db).unwrap();
    let traces: i64 = conn
        .query_row("SELECT count(*) FROM traces", [], |r| r.get(0))
        .unwrap();
    assert_eq!(traces, 4);
    let jobs: Vec<(String, String)> = conn
        .prepare("SELECT j.evaluator, t.tags FROM eval_jobs j JOIN traces t ON t.id = j.trace_id")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        jobs,
        [("tagged".to_owned(), r#"{"project":"ig"}"#.to_owned())]
    );
}

const TOOL_USE_SSE: &[u8] = include_bytes!("fixtures/anthropic/cc_tool_use.sse");

/// Jobs per stop reason, live and by backfill.
#[tokio::test]
async fn evaluators_select_by_stop_reason() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("g.db");
    let text = MockUpstream::start(Script::sse(TEXT_SSE, Duration::ZERO)).await;
    let tools = MockUpstream::start(Script::sse(TOOL_USE_SSE, Duration::ZERO)).await;
    let evaluators = r#"
[[eval.evaluators]]
name = "answer"
match = { stop_reason = ["end_turn"] }
state = "{{ output_text }}"
question = { type = "noul", instructions = "?" }

[[eval.evaluators]]
name = "tool"
match = { stop_reason = ["tool_use"] }
state = "{{ tool_calls }}"
question = { type = "noul", instructions = "?" }
"#;
    let section = |evaluators: &str| {
        format!(
            "[store]\npath = \"{}\"\n\n[capture]\n\n[[capture.rules]]\n\n[eval]\npoll = \"1h\"\n\n[eval.judge]\nbase_url = \"http://{}\"\nmodel = \"jev-1.13.0\"\napi_key_env = \"IG_TEST_UNUSED\"\n{evaluators}",
            db.display(),
            unreachable_addr()
        )
    };
    let gw = spawn_gateway_with(
        &config(
            &[
                ("text", "anthropic", &text.base_url()),
                ("tools", "anthropic", &tools.base_url()),
            ],
            &section(evaluators),
        ),
        Options {
            judge_key: Some("k".into()),
            ..Options::default()
        },
    )
    .await;
    let body =
        r#"{"model":"claude-sonnet-5","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    for upstream in ["text", "tools"] {
        client()
            .post(gw.url(&format!("/{upstream}/v1/messages")))
            .body(body)
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
    }
    gw.shutdown(Duration::from_secs(5)).await.unwrap();

    let mut conn = open_writer(&db).unwrap();
    let jobs = |conn: &Connection| -> Vec<(String, String)> {
        conn.prepare(
            "SELECT j.evaluator, t.stop_reason FROM eval_jobs j JOIN traces t ON t.id = j.trace_id
             ORDER BY j.evaluator, t.stop_reason",
        )
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
    };
    let pairs = |p: &[(&str, &str)]| -> Vec<(String, String)> {
        p.iter()
            .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
            .collect()
    };
    assert_eq!(
        jobs(&conn),
        pairs(&[("answer", "end_turn"), ("tool", "tool_use")])
    );

    // A new version of `tool` that takes both stop reasons: backfill finds both.
    let cfg = inference_gateway::config::Config::parse(&config(
        &[("text", "anthropic", "http://x")],
        &section(&evaluators.replace(
            "match = { stop_reason = [\"tool_use\"] }\nstate = \"{{ tool_calls }}\"",
            "match = { stop_reason = [\"tool_use\", \"end_turn\"] }\nstate = \"{{ stop_reason }} {{ tool_calls }}\"",
        )),
    ))
    .unwrap();
    let tool = &cfg.eval.unwrap().evaluators[1];
    let r =
        inference_gateway::eval::backfill::backfill(&mut conn, tool, 0, false, false, 0).unwrap();
    assert_eq!((r.matched, r.enqueued), (2, 2));
}

/// Claude Code side calls ask for JSON output; `structured_output` tells
/// them apart, live and by backfill.
#[tokio::test]
async fn evaluators_select_by_structured_output() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("g.db");
    let upstream = MockUpstream::start(Script::sse(TEXT_SSE, Duration::ZERO)).await;
    let evaluators = r#"
[[eval.evaluators]]
name = "plain"
match = { structured_output = false }
state = "{{ output_text }}"
question = { type = "noul", instructions = "?" }

[[eval.evaluators]]
name = "json"
match = { structured_output = true }
state = "{{ output_text }}"
question = { type = "noul", instructions = "?" }
"#;
    let section = |evaluators: &str| {
        format!(
            "[store]\npath = \"{}\"\n\n[capture]\n\n[[capture.rules]]\n\n[eval]\npoll = \"1h\"\n\n[eval.judge]\nbase_url = \"http://{}\"\nmodel = \"jev-1.13.0\"\napi_key_env = \"IG_TEST_UNUSED\"\n{evaluators}",
            db.display(),
            unreachable_addr()
        )
    };
    let gw = spawn_gateway_with(
        &config(
            &[("anthropic", "anthropic", &upstream.base_url())],
            &section(evaluators),
        ),
        Options {
            judge_key: Some("k".into()),
            ..Options::default()
        },
    )
    .await;
    for body in [
        r#"{"model":"claude-sonnet-5","stream":true,"messages":[{"role":"user","content":"hi"}]}"#,
        r#"{"model":"claude-sonnet-5","stream":true,"output_config":{"format":{"type":"json_schema","schema":{}}},"messages":[{"role":"user","content":"title?"}]}"#,
    ] {
        client()
            .post(gw.url("/anthropic/v1/messages"))
            .body(body)
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
    }
    gw.shutdown(Duration::from_secs(5)).await.unwrap();

    let mut conn = open_writer(&db).unwrap();
    let jobs: Vec<(String, bool)> = conn
        .prepare(
            "SELECT j.evaluator, p.json LIKE '%json_schema%'
               FROM eval_jobs j JOIN trace_params p ON p.trace_id = j.trace_id
              ORDER BY j.evaluator",
        )
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        jobs,
        [("json".to_owned(), true), ("plain".to_owned(), false)]
    );

    // A new version of `plain`: backfill skips the JSON request too.
    let cfg = inference_gateway::config::Config::parse(&config(
        &[("anthropic", "anthropic", "http://x")],
        &section(&evaluators.replacen("{{ output_text }}", "v2 {{ output_text }}", 1)),
    ))
    .unwrap();
    let plain = &cfg.eval.unwrap().evaluators[0];
    let r =
        inference_gateway::eval::backfill::backfill(&mut conn, plain, 0, false, false, 0).unwrap();
    assert_eq!((r.matched, r.enqueued), (1, 1));
}
