#![cfg(feature = "otel")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::collections::BTreeMap;
use std::time::Duration;

use axum::http::StatusCode;
use serde_json::Value;
use tokio::time::Instant;

use support::mock_upstream::{MockUpstream, Script, unreachable_addr};
use support::{TestGateway, client, config, spawn_gateway};

const TEXT_SSE: &[u8] = include_bytes!("fixtures/anthropic/text.sse");
const MESSAGES: &str = r#"{"model":"claude-sonnet-5","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
const TRACEPARENT: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
const GRACE: Duration = Duration::from_secs(5);

async fn collector() -> MockUpstream {
    MockUpstream::start(Script::json(StatusCode::OK, b"{}")).await
}

async fn gateway(upstream: &MockUpstream, otel: &str, extra: &str) -> TestGateway {
    spawn_gateway(
        &config(
            &[("anthropic", "anthropic", &upstream.base_url())],
            &format!("[otel]\n{otel}\n{extra}"),
        ),
        &[],
        None,
    )
    .await
}

async fn post(gw: &TestGateway, traceparent: Option<&str>) {
    let mut req = client()
        .post(gw.url("/anthropic/v1/messages"))
        .header("x-api-key", "sk-test")
        .header("x-gateway-tag-project", "ig")
        .body(MESSAGES);
    if let Some(tp) = traceparent {
        req = req.header("traceparent", tp);
    }
    let resp = req.send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    resp.bytes().await.unwrap();
}

/// Every span the collector received, in arrival order.
fn spans(collector: &MockUpstream) -> Vec<Value> {
    collector
        .requests()
        .iter()
        .flat_map(|r| {
            assert_eq!(r.uri.path(), "/v1/traces");
            assert_eq!(r.headers["content-type"], "application/json");
            let body: Value = serde_json::from_slice(&r.body).unwrap();
            body["resourceSpans"][0]["scopeSpans"][0]["spans"]
                .as_array()
                .unwrap()
                .clone()
        })
        .collect()
}

async fn wait_spans(collector: &MockUpstream, n: usize) -> Vec<Value> {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let spans = spans(collector);
        if spans.len() >= n || Instant::now() > deadline {
            return spans;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn attrs(span: &Value) -> BTreeMap<String, Value> {
    span["attributes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| (a["key"].as_str().unwrap().to_owned(), a["value"].clone()))
        .collect()
}

#[tokio::test]
async fn span_is_exported_with_the_record_id() {
    let upstream = MockUpstream::start(Script::sse(TEXT_SSE, Duration::ZERO)).await;
    let collector = collector().await;
    let otel = format!("endpoint = \"{}\"\nflush = \"20ms\"", collector.base_url());
    let mut gw = gateway(&upstream, &otel, "").await;
    post(&gw, None).await;
    let r = gw.record().await;

    let spans = wait_spans(&collector, 1).await;
    assert_eq!(spans.len(), 1);
    let s = &spans[0];
    assert_eq!(s["traceId"], r.id.simple().to_string());
    assert!(s.get("parentSpanId").is_none());
    assert_eq!(s["name"], "chat claude-sonnet-5");
    let a = attrs(s);
    assert_eq!(a["gen_ai.provider.name"]["stringValue"], "anthropic");
    assert_eq!(
        a["gen_ai.usage.output_tokens"]["intValue"],
        r.usage.output_tokens.unwrap().to_string()
    );
    assert_eq!(a["gateway.tag.project"]["stringValue"], "ig");
    let events: Vec<_> = s["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert_eq!(events, ["first_byte", "first_token"]);
    let nanos = |k: &str| s[k].as_str().unwrap().parse::<u128>().unwrap();
    assert!(nanos("startTimeUnixNano") < nanos("endTimeUnixNano"));

    // No content and no keys in the span.
    let text = s.to_string();
    assert!(!text.contains("sk-test"));
    assert!(!text.contains(r#""hi""#));
    gw.shutdown(GRACE).await.unwrap();
}

#[cfg(feature = "store")]
#[tokio::test]
async fn traceparent_joins_the_callers_trace() {
    let upstream = MockUpstream::start(Script::sse(TEXT_SSE, Duration::ZERO)).await;
    let collector = collector().await;
    let dir = tempfile::tempdir().unwrap();
    let otel = format!("endpoint = \"{}\"\nflush = \"20ms\"", collector.base_url());
    let store = format!(
        "[store]\npath = \"{}\"\n",
        dir.path().join("gateway.db").display()
    );
    let mut gw = gateway(&upstream, &otel, &store).await;
    post(&gw, Some(TRACEPARENT)).await;
    let r = gw.record().await;
    assert_eq!(upstream.last_request().headers["traceparent"], TRACEPARENT);

    let spans = wait_spans(&collector, 1).await;
    assert_eq!(spans[0]["traceId"], "4bf92f3577b34da6a3ce929d0e0e4736");
    assert_eq!(spans[0]["parentSpanId"], "00f067aa0ba902b7");
    assert_eq!(
        attrs(&spans[0])["gateway.trace_id"]["stringValue"],
        r.id.simple().to_string()
    );

    let store = gw.ctx.store.clone().unwrap();
    gw.shutdown(GRACE).await.unwrap();
    let stored: Vec<u8> = store
        .reader()
        .unwrap()
        .query_row("SELECT otel_trace_id FROM traces", [], |row| row.get(0))
        .unwrap();
    assert_eq!(hex::encode(stored), "4bf92f3577b34da6a3ce929d0e0e4736");
}

#[tokio::test]
async fn unsampled_requests_export_nothing() {
    let upstream = MockUpstream::start(Script::sse(TEXT_SSE, Duration::ZERO)).await;
    let collector = collector().await;

    // The caller's decision wins over sample_ratio.
    let otel = format!("endpoint = \"{}\"", collector.base_url());
    let gw = gateway(&upstream, &otel, "").await;
    post(&gw, Some(&TRACEPARENT.replace("-01", "-00"))).await;
    gw.shutdown(GRACE).await.unwrap();

    let otel = format!(
        "endpoint = \"{}\"\nsample_ratio = 0.0",
        collector.base_url()
    );
    let gw = gateway(&upstream, &otel, "").await;
    post(&gw, None).await;
    gw.shutdown(GRACE).await.unwrap();

    assert!(collector.requests().is_empty());
}

#[tokio::test]
async fn shutdown_flushes_pending_spans() {
    let upstream = MockUpstream::start(Script::sse(TEXT_SSE, Duration::ZERO)).await;
    let collector = collector().await;
    let otel = format!("endpoint = \"{}\"\nflush = \"1h\"", collector.base_url());
    let mut gw = gateway(&upstream, &otel, "").await;
    for _ in 0..3 {
        post(&gw, None).await;
        gw.record().await;
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(collector.requests().is_empty());
    gw.shutdown(GRACE).await.unwrap();
    assert_eq!(spans(&collector).len(), 3);
    assert_eq!(collector.requests().len(), 1);
}

#[tokio::test]
async fn unreachable_collector_does_not_affect_the_proxy() {
    let upstream = MockUpstream::start(Script::sse(TEXT_SSE, Duration::ZERO)).await;
    let otel = format!(
        "endpoint = \"http://{}\"\nflush = \"10ms\"\ntimeout = \"200ms\"",
        unreachable_addr()
    );
    let mut gw = gateway(&upstream, &otel, "").await;
    for _ in 0..3 {
        post(&gw, None).await;
        assert_eq!(gw.record().await.outcome.as_str(), "ok");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    gw.shutdown(GRACE).await.unwrap();
}
