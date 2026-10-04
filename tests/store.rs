#![cfg(feature = "store")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::time::{Duration, UNIX_EPOCH};

use inference_gateway::store::Store;
use rusqlite::Connection;
use serde_json::{Value, json};

use support::mock_upstream::{MockUpstream, Script};
use support::{TestGateway, client, config, spawn_gateway};

const TEXT_SSE: &[u8] = include_bytes!("fixtures/anthropic/text.sse");
const MESSAGES: &str = r#"{"model":"claude-sonnet-5","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
const GRACE: Duration = Duration::from_secs(5);

struct Setup {
    gw: TestGateway,
    store: Store,
    dir: tempfile::TempDir,
}

/// Gateway with `[store]` and `capture` appended verbatim.
async fn setup(mock: &MockUpstream, capture: &str) -> Setup {
    let dir = tempfile::tempdir().unwrap();
    let extra = format!(
        "[store]\npath = \"{}\"\n\n{capture}",
        dir.path().join("gateway.db").display()
    );
    let gw = spawn_gateway(
        &config(&[("anthropic", "anthropic", &mock.base_url())], &extra),
        &[],
        None,
    )
    .await;
    let store = gw.ctx.store.clone().expect("store module built");
    Setup { gw, store, dir }
}

async fn post(gw: &TestGateway, body: &str) -> Vec<u8> {
    client()
        .post(gw.url("/anthropic/v1/messages"))
        .header("x-api-key", "sk-test")
        .header("x-gateway-session", "run-7")
        .header("x-gateway-tag-project", "ig")
        .body(body.to_owned())
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap()
        .to_vec()
}

/// Shutdown drains the writer, so every trace is committed afterwards.
async fn finish(s: Setup) -> (Store, Connection, tempfile::TempDir) {
    s.gw.shutdown(GRACE).await.unwrap();
    let conn = s.store.reader().unwrap();
    (s.store, conn, s.dir)
}

fn count(conn: &Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |r| r.get(0)).unwrap()
}

#[tokio::test]
async fn trace_row_matches_record() {
    let mock = MockUpstream::start(Script::sse(TEXT_SSE, Duration::ZERO)).await;
    let mut s = setup(&mock, "").await;
    post(&s.gw, MESSAGES).await;
    let r = s.gw.record().await;
    let (_store, conn, _dir) = finish(s).await;

    assert_eq!(count(&conn, "SELECT count(*) FROM traces"), 1);
    let row = conn
        .query_row(
            "SELECT started_at, upstream, provider, method, path, operation, requested_model,
                    response_model, streaming, session_id, tags, http_status, outcome, error_type,
                    stop_reason, input_tokens_total, output_tokens, duration_ms, ttfb_ms,
                    chunk_count, ratelimit_requests_remaining, cost_usd, captured
             FROM traces WHERE id = ?1",
            [r.id.as_bytes().as_slice()],
            |row| {
                Ok((
                    (
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, Option<String>>(5)?,
                        row.get::<_, Option<String>>(6)?,
                        row.get::<_, Option<String>>(7)?,
                    ),
                    (
                        row.get::<_, bool>(8)?,
                        row.get::<_, Option<String>>(9)?,
                        row.get::<_, String>(10)?,
                        row.get::<_, Option<u16>>(11)?,
                        row.get::<_, String>(12)?,
                        row.get::<_, Option<String>>(13)?,
                        row.get::<_, Option<String>>(14)?,
                    ),
                    (
                        row.get::<_, Option<u64>>(15)?,
                        row.get::<_, Option<u64>>(16)?,
                        row.get::<_, f64>(17)?,
                        row.get::<_, Option<f64>>(18)?,
                        row.get::<_, u64>(19)?,
                        row.get::<_, Option<u64>>(20)?,
                        row.get::<_, Option<f64>>(21)?,
                        row.get::<_, bool>(22)?,
                    ),
                ))
            },
        )
        .unwrap();

    let started_ms = r.started_at.duration_since(UNIX_EPOCH).unwrap().as_millis();
    assert_eq!(
        row.0,
        (
            i64::try_from(started_ms).unwrap(),
            r.upstream.clone(),
            r.provider.as_str().to_owned(),
            r.method.clone(),
            r.path.clone(),
            r.operation.map(|o| o.as_str().to_owned()),
            r.requested_model.clone(),
            r.response_model.clone(),
        )
    );
    assert_eq!(
        row.1,
        (
            r.streaming,
            r.principal.session_id.clone(),
            r#"{"project":"ig"}"#.to_owned(),
            r.http_status,
            r.outcome.as_str().to_owned(),
            r.error_type.clone(),
            r.stop_reason.clone(),
        )
    );
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    assert_eq!(
        row.2,
        (
            r.usage.input_tokens_total,
            r.usage.output_tokens,
            ms(r.timings.duration),
            r.timings.ttfb.map(ms),
            r.timings.chunk_count,
            r.rate_limits.requests_remaining,
            r.cost_usd,
            false,
        )
    );
    assert_eq!(r.rate_limits.requests_remaining, Some(49));
    for table in ["trace_inputs", "trace_params", "trace_outputs", "blobs"] {
        assert_eq!(count(&conn, &format!("SELECT count(*) FROM {table}")), 0);
    }
}

/// Claude Code style: full history every turn, cache breakpoint on the last message.
fn turn_body(history: &[Value]) -> String {
    let mut messages = history.to_vec();
    if let Some(Value::Array(blocks)) = messages.last_mut().unwrap().get_mut("content") {
        blocks.last_mut().unwrap()["cache_control"] = json!({"type": "ephemeral"});
    }
    json!({
        "model": "claude-sonnet-5",
        "max_tokens": 64,
        "stream": true,
        "system": [{"type": "text", "text": "You are a coding agent. ".repeat(200), "cache_control": {"type": "ephemeral"}}],
        "tools": [{"name": "Bash", "description": "Run a command", "input_schema": {"type": "object"}}],
        "messages": messages,
    })
    .to_string()
}

fn user(text: &str) -> Value {
    json!({"role": "user", "content": [{"type": "text", "text": text}]})
}

#[tokio::test]
async fn resent_history_is_stored_once() {
    let mock = MockUpstream::start(Script::sse(TEXT_SSE, Duration::ZERO)).await;
    let mut s = setup(&mock, "[capture]\n\n[[capture.rules]]\n").await;

    let mut history = Vec::new();
    let mut last = None;
    for turn in 1..=5 {
        history.push(user(&format!("question {turn}")));
        post(&s.gw, &turn_body(&history)).await;
        last = Some(s.gw.record().await.id);
        history.push(json!({"role": "assistant", "content": [{"type": "text", "text": format!("answer {turn}")}]}));
    }
    let (store, conn, _dir) = finish(s).await;

    // Turn k adds k-th user message plus the (k-1)-th answer.
    let ids: Vec<Vec<u8>> = conn
        .prepare("SELECT id FROM traces ORDER BY started_at, id")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(ids.len(), 5);
    for k in 1..=5 {
        let distinct: i64 = conn
            .query_row(
                "SELECT count(DISTINCT blob_hash) FROM trace_inputs WHERE trace_id IN
                 (SELECT id FROM traces ORDER BY started_at, id LIMIT ?1)",
                [k],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(distinct, 2 + 2 * k - 1, "after turn {k}");
    }
    assert_eq!(
        count(&conn, "SELECT count(*) FROM trace_inputs"),
        5 * 2 + 25
    );
    // Identical responses share one raw blob.
    assert_eq!(count(&conn, "SELECT count(*) FROM blobs"), 2 + 9 + 1);
    assert_eq!(
        count(
            &conn,
            "SELECT count(*) FROM trace_inputs WHERE kind = 'system'"
        ),
        5
    );

    let content = store.load_trace_content(last.unwrap()).unwrap().unwrap();
    history.pop();
    assert_eq!(content.request.messages, history);
    assert_eq!(content.request.tools.unwrap()[0]["name"], "Bash");
    let system = content.request.system.unwrap();
    assert!(system[0].get("cache_control").is_none());
    assert_eq!(
        content.request.params,
        *json!({"model": "claude-sonnet-5", "max_tokens": 64, "stream": true})
            .as_object()
            .unwrap()
    );
    assert_eq!(content.raw_response, TEXT_SSE);
    assert_eq!(
        content.output_text.as_deref(),
        Some("Hello! How can I help you today?")
    );
    assert_eq!(content.last_user_text.as_deref(), Some("question 5"));
    assert!(!content.truncated);
}

#[tokio::test]
async fn oversized_response_is_stored_truncated() {
    let mock = MockUpstream::start(Script::sse(TEXT_SSE, Duration::ZERO)).await;
    let mut s = setup(
        &mock,
        "[capture]\nmax_bytes = \"100 B\"\n\n[[capture.rules]]\n",
    )
    .await;
    assert_eq!(post(&s.gw, MESSAGES).await, TEXT_SSE);
    let id = s.gw.record().await.id;
    let (store, conn, _dir) = finish(s).await;

    assert_eq!(
        count(&conn, "SELECT truncated FROM trace_outputs"),
        1,
        "truncated flag"
    );
    let content = store.load_trace_content(id).unwrap().unwrap();
    assert_eq!(content.raw_response, &TEXT_SSE[..100]);
    assert!(content.truncated);
}

#[tokio::test]
async fn capture_off_stores_trace_only() {
    let mock = MockUpstream::start(Script::sse(TEXT_SSE, Duration::ZERO)).await;
    let mut s = setup(&mock, "[capture]\n\n[[capture.rules]]\ncapture = false\n").await;
    post(&s.gw, MESSAGES).await;
    let id = s.gw.record().await.id;
    let (store, conn, _dir) = finish(s).await;

    assert_eq!(
        count(&conn, "SELECT count(*) FROM traces WHERE captured = 0"),
        1
    );
    for table in ["trace_inputs", "trace_params", "trace_outputs", "blobs"] {
        assert_eq!(count(&conn, &format!("SELECT count(*) FROM {table}")), 0);
    }
    assert_eq!(store.load_trace_content(id).unwrap(), None);
}

#[tokio::test]
async fn unparsable_request_keeps_trace_and_output() {
    let mock = MockUpstream::start(Script::sse(TEXT_SSE, Duration::ZERO)).await;
    let mut s = setup(&mock, "[capture]\n\n[[capture.rules]]\n").await;
    post(&s.gw, "not json").await;
    let id = s.gw.record().await.id;
    let (store, conn, _dir) = finish(s).await;

    assert_eq!(
        count(&conn, "SELECT count(*) FROM traces WHERE captured = 1"),
        1
    );
    assert_eq!(count(&conn, "SELECT count(*) FROM trace_inputs"), 0);
    let content = store.load_trace_content(id).unwrap().unwrap();
    assert_eq!(content.raw_response, TEXT_SSE);
    assert_eq!(content.last_user_text, None);
}
