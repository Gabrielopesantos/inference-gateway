//! Load checks with `oha` (in the dev shell). Ignored by default:
//!
//!     cargo test --release --test load -- --ignored --nocapture --test-threads 1

#![cfg(feature = "eval")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::path::Path;
use std::process::Command;
use std::time::Duration;

use axum::http::StatusCode;
use inference_gateway::modules::DropCounters;
use inference_gateway::store::open_reader;
use serde_json::Value;

use support::mock_upstream::{MockUpstream, Script};
use support::{Options, TestGateway, config, spawn_gateway_with};

const TEXT_SSE: &[u8] = include_bytes!("fixtures/anthropic/text.sse");
const CHOICE: &[u8] = include_bytes!("fixtures/judge/choice.json");
const MESSAGES: &str = r#"{"model":"claude-sonnet-5","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi there"}]}"#;
const CONNECTIONS: usize = 200;
const REQUESTS: usize = 4000;
const WARMUP: usize = 400;
const GRACE: Duration = Duration::from_secs(30);

struct Stats {
    p50_ms: f64,
    p99_ms: f64,
    ok: u64,
}

/// Runs `oha` against the gateway and parses its JSON report.
async fn oha(gw: &TestGateway, requests: usize) -> Stats {
    let url = gw.url("/anthropic/v1/messages");
    let out = tokio::task::spawn_blocking(move || {
        Command::new("oha")
            .args(["--no-tui", "--output-format", "json"])
            .args(["-n", &requests.to_string()])
            .args(["-c", &CONNECTIONS.to_string()])
            .args(["-m", "POST", "-T", "application/json", "-d", MESSAGES])
            .args(["-H", "x-api-key: sk-test"])
            .arg(url)
            .output()
            .expect("oha not found; run inside `nix develop`")
    })
    .await
    .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: Value = serde_json::from_slice(&out.stdout).unwrap();
    let ms = |p: &str| {
        report["latencyPercentiles"][p]
            .as_f64()
            .unwrap_or_else(|| panic!("no {p} in {report}"))
            * 1000.0
    };
    Stats {
        p50_ms: ms("p50"),
        p99_ms: ms("p99"),
        ok: report["statusCodeDistribution"]["200"]
            .as_u64()
            .unwrap_or(0),
    }
}

/// `[store]` with every request captured, plus `extra`.
async fn gateway(db: &Path, upstream: &MockUpstream, extra: &str) -> TestGateway {
    let extra = format!(
        "[prometheus]\n\n[store]\npath = \"{}\"\n\n[capture]\n\n[[capture.rules]]\n\n{extra}",
        db.display()
    );
    spawn_gateway_with(
        &config(&[("anthropic", "anthropic", &upstream.base_url())], &extra),
        Options {
            judge_key: Some("k".into()),
            ..Options::default()
        },
    )
    .await
}

fn count(db: &Path, sql: &str) -> u64 {
    open_reader(db)
        .unwrap()
        .query_row(sql, [], |r| r.get(0))
        .unwrap()
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "p50={:.1}ms p99={:.1}ms ok={}",
            self.p50_ms, self.p99_ms, self.ok
        )
    }
}

fn store_drops(drops: &DropCounters) -> u64 {
    drops
        .snapshot()
        .into_iter()
        .find(|(name, _)| *name == "store")
        .map_or(0, |(_, n)| n)
}

async fn upstream() -> MockUpstream {
    MockUpstream::start(Script::sse(TEXT_SSE, Duration::from_millis(5))).await
}

/// Every exchange is stored, or its drop is counted.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "load test; needs oha"]
async fn capture_under_load() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("g.db");
    let up = upstream().await;
    let gw = gateway(&db, &up, "").await;
    let stats = oha(&gw, REQUESTS).await;
    let drops = gw.ctx.drops.clone();
    gw.shutdown(GRACE).await.unwrap();
    let dropped = store_drops(&drops);
    let traces = count(&db, "SELECT count(*) FROM traces");
    let outputs = count(&db, "SELECT count(*) FROM trace_outputs");
    println!("capture_under_load: {stats} traces={traces} captured={outputs} dropped={dropped}");
    assert_eq!(stats.ok, REQUESTS as u64);
    assert_eq!(traces + dropped, REQUESTS as u64);
    assert_eq!(outputs, traces);
}

/// One measured run after a warm-up, on a fresh database.
async fn measure(eval: bool) -> (Stats, u64, u64) {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("g.db");
    let up = upstream().await;
    let mut slow = Script::json(StatusCode::OK, CHOICE);
    slow.header_delay = Duration::from_millis(200);
    let judge = MockUpstream::start(slow).await;
    let section = format!(
        r#"
[eval.worker]
concurrency = 4
poll = "50ms"

[eval.judge]
base_url = "{}"
model = "jev-1.13.0"
api_key_env = "IG_TEST_UNUSED"

[[eval.evaluators]]
name = "q"
input = "{{{{ last_user_message }}}} {{{{ output_text }}}}"

kind = "choice"
question = "Did the response address the request?"
criteria = {{ correct = "Fully answers", partial = "Answers part", wrong = "" }}
"#,
        judge.base_url()
    );
    let gw = gateway(&db, &up, if eval { &section } else { "" }).await;
    oha(&gw, WARMUP).await;
    let stats = oha(&gw, REQUESTS).await;
    assert_eq!(stats.ok, REQUESTS as u64);
    let dropped = store_drops(&gw.ctx.drops);
    let scored = count(&db, "SELECT count(*) FROM scores");
    let queued = count(
        &db,
        "SELECT count(*) FROM eval_jobs WHERE status IN ('pending', 'running')",
    );
    gw.shutdown(GRACE).await.unwrap();
    assert_eq!(dropped, 0, "store dropped records");
    (stats, scored, queued)
}

/// A busy eval queue does not slow the proxy.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "load test; needs oha"]
async fn proxy_latency_with_eval_busy() {
    let (base, _, _) = measure(false).await;
    let (busy, scored, queued) = measure(true).await;
    println!("without eval: {base}");
    println!("with eval:    {busy} scored={scored} still_queued={queued}");
    assert!(scored > 0, "worker scored nothing");
    assert!(queued > 0, "queue drained; workers were not busy");
    assert!(
        busy.p99_ms <= base.p99_ms * 1.25 + 5.0,
        "p99 {} ms with eval vs {} ms without",
        busy.p99_ms,
        base.p99_ms
    );
}
