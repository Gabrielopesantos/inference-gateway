#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::time::Duration;

use axum::http::StatusCode;
use inference_gateway::observe::record::Outcome;
use inference_gateway::providers::Operation;

use support::mock_upstream::{MockUpstream, Script, unreachable_addr};
use support::{TestGateway, client, config, spawn_gateway};

const ANTHROPIC_SSE: &[u8] = include_bytes!("fixtures/anthropic/text.sse");
const ANTHROPIC_ERROR_SSE: &[u8] = include_bytes!("fixtures/anthropic/error.sse");
const ANTHROPIC_JSON: &[u8] = include_bytes!("fixtures/anthropic/text.json");
const ANTHROPIC_429: &[u8] = include_bytes!("fixtures/anthropic/rate_limited.json");
const ANTHROPIC_429_HEADERS: &str = include_str!("fixtures/anthropic/rate_limited.headers");
const OPENAI_SSE: &[u8] = include_bytes!("fixtures/openai/text.sse");

const MESSAGES: &str = r#"{"model":"claude-sonnet-5","max_tokens":64,"stream":true,"messages":[]}"#;

async fn gateway(provider: &str, mock: &MockUpstream, flags: &[&str]) -> TestGateway {
    spawn_gateway(
        &config(&[(provider, provider, &mock.base_url())], ""),
        flags,
        None,
    )
    .await
}

async fn post(gw: &TestGateway, path: &str, body: &str) -> reqwest::Response {
    client()
        .post(gw.url(path))
        .body(body.to_owned())
        .send()
        .await
        .unwrap()
}

#[tokio::test]
async fn stream_produces_one_record() {
    let mock = MockUpstream::start(Script::sse(ANTHROPIC_SSE, Duration::from_millis(2))).await;
    let mut gw = gateway("anthropic", &mock, &[]).await;
    post(&gw, "/anthropic/v1/messages", MESSAGES)
        .await
        .bytes()
        .await
        .unwrap();

    let r = gw.record().await;
    assert_eq!(r.outcome, Outcome::Ok);
    assert_eq!(r.upstream, "anthropic");
    assert_eq!(r.operation, Some(Operation::Messages));
    assert_eq!(r.requested_model.as_deref(), Some("claude-sonnet-5"));
    assert_eq!(r.response_model.as_deref(), Some("claude-sonnet-5"));
    assert_eq!(r.http_status, Some(200));
    assert!(r.streaming);
    assert_eq!(r.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(r.usage.input_tokens_total, Some(25));
    assert_eq!(r.usage.output_tokens, Some(12));
    assert_eq!(r.rate_limits.requests_remaining, Some(49));
    assert!(r.timings.ttft.is_some());
    assert!(r.timings.ttfb <= r.timings.ttft);
    gw.assert_no_record(Duration::from_millis(100)).await;
}

#[tokio::test]
async fn client_cancel_produces_one_cancelled_record() {
    let script = Script::chunks(vec![
        (Duration::ZERO, "event: ping\ndata: {\"type\":\"ping\"}\n\n"),
        (Duration::from_secs(10), "data: never\n\n"),
    ]);
    let mock = MockUpstream::start(script).await;
    let mut gw = gateway("anthropic", &mock, &[]).await;
    let mut resp = post(&gw, "/anthropic/v1/messages", MESSAGES).await;
    resp.chunk().await.unwrap().unwrap();
    drop(resp);

    let r = gw.record().await;
    assert_eq!(r.outcome, Outcome::ClientCancelled);
    assert_eq!(r.timings.chunk_count, 1);
    gw.assert_no_record(Duration::from_millis(100)).await;
}

#[tokio::test]
async fn mid_stream_error_event_is_upstream_error() {
    let mock =
        MockUpstream::start(Script::sse(ANTHROPIC_ERROR_SSE, Duration::from_millis(1))).await;
    let mut gw = gateway("anthropic", &mock, &[]).await;
    post(&gw, "/anthropic/v1/messages", MESSAGES)
        .await
        .bytes()
        .await
        .unwrap();

    let r = gw.record().await;
    assert_eq!(r.outcome, Outcome::UpstreamError);
    assert_eq!(r.error_type.as_deref(), Some("overloaded_error"));
    assert_eq!(r.http_status, Some(200));
}

#[tokio::test]
async fn rate_limited_records_headroom() {
    let script = Script::json(StatusCode::TOO_MANY_REQUESTS, ANTHROPIC_429)
        .with_fixture_headers(ANTHROPIC_429_HEADERS);
    let mock = MockUpstream::start(script).await;
    let mut gw = gateway("anthropic", &mock, &[]).await;
    let resp = post(&gw, "/anthropic/v1/messages", MESSAGES).await;
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    resp.bytes().await.unwrap();

    let r = gw.record().await;
    assert_eq!(r.outcome, Outcome::UpstreamError);
    assert_eq!(r.error_type.as_deref(), Some("rate_limit_error"));
    assert_eq!(r.http_status, Some(429));
    assert_eq!(r.rate_limits.requests_remaining, Some(0));
    assert_eq!(r.rate_limits.tokens_remaining, Some(81000));
    assert_eq!(r.rate_limits.retry_after, Some(Duration::from_secs(12)));
}

#[tokio::test]
async fn non_stream_json_is_parsed() {
    let mock = MockUpstream::start(Script::json(StatusCode::OK, ANTHROPIC_JSON)).await;
    let mut gw = gateway("anthropic", &mock, &[]).await;
    post(
        &gw,
        "/anthropic/v1/messages",
        r#"{"model":"claude-sonnet-5","max_tokens":64,"messages":[]}"#,
    )
    .await
    .bytes()
    .await
    .unwrap();

    let r = gw.record().await;
    assert_eq!(r.outcome, Outcome::Ok);
    assert!(!r.streaming);
    assert_eq!(r.usage.output_tokens, Some(12));
    assert_eq!(r.stop_reason.as_deref(), Some("end_turn"));
}

#[tokio::test]
async fn openai_stream_gets_include_usage() {
    let mock = MockUpstream::start(Script::sse(OPENAI_SSE, Duration::from_millis(1))).await;
    let mut gw = gateway("openai", &mock, &[]).await;
    post(
        &gw,
        "/openai/v1/chat/completions",
        r#"{"model":"gpt-5-mini","stream":true,"messages":[]}"#,
    )
    .await
    .bytes()
    .await
    .unwrap();

    let sent: serde_json::Value = serde_json::from_slice(&mock.last_request().body).unwrap();
    assert_eq!(sent["stream_options"]["include_usage"], true);
    let r = gw.record().await;
    assert_eq!(r.operation, Some(Operation::ChatCompletions));
    assert_eq!(r.usage.input_tokens_total, Some(19));
    assert_eq!(r.usage.output_tokens, Some(10));
}

#[tokio::test]
async fn unobserved_path_still_records() {
    let mock = MockUpstream::start(Script::json(StatusCode::OK, b"{\"data\":[]}")).await;
    let mut gw = gateway("openai", &mock, &[]).await;
    client()
        .get(gw.url("/openai/v1/models"))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();

    let r = gw.record().await;
    assert_eq!(r.operation, None);
    assert_eq!(r.outcome, Outcome::Ok);
    assert_eq!(r.usage.input_tokens_total, None);
}

#[tokio::test]
async fn unreachable_upstream_is_gateway_error() {
    let base = format!("http://{}", unreachable_addr());
    let mut gw = spawn_gateway(&config(&[("openai", "openai", &base)], ""), &[], None).await;
    let resp = post(&gw, "/openai/v1/chat/completions", "{}").await;
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

    let r = gw.record().await;
    assert_eq!(r.outcome, Outcome::GatewayError);
    assert_eq!(
        r.error_type.as_deref(),
        Some("gateway_upstream_unreachable")
    );
    assert_eq!(r.http_status, Some(502));
}

#[tokio::test]
async fn oversized_body_is_gateway_error() {
    let mock = MockUpstream::start(Script::json(StatusCode::OK, b"{}")).await;
    let mut gw = gateway("anthropic", &mock, &["--max-body", "1KiB"]).await;
    let resp = post(&gw, "/anthropic/v1/messages", &"x".repeat(2048)).await;
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);

    let r = gw.record().await;
    assert_eq!(r.outcome, Outcome::GatewayError);
    assert_eq!(r.error_type.as_deref(), Some("gateway_body_too_large"));
}

#[tokio::test]
async fn unknown_upstream_has_no_record() {
    let mock = MockUpstream::start(Script::json(StatusCode::OK, b"{}")).await;
    let mut gw = gateway("anthropic", &mock, &[]).await;
    let resp = post(&gw, "/nope/v1/messages", MESSAGES).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    gw.assert_no_record(Duration::from_millis(100)).await;
}

#[tokio::test]
async fn head_response_is_ok_not_cancelled() {
    let mock = MockUpstream::start(Script::json(StatusCode::OK, b"{}")).await;
    let mut gw = gateway("anthropic", &mock, &[]).await;
    let resp = client().head(gw.url("/anthropic/")).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    resp.bytes().await.unwrap();

    let r = gw.record().await;
    assert_eq!(r.outcome, Outcome::Ok);
    assert_eq!(r.method, "HEAD");
    assert_eq!(r.path, "/");
    gw.assert_no_record(Duration::from_millis(100)).await;
}

#[tokio::test]
async fn no_content_response_is_ok_not_cancelled() {
    let script = Script {
        status: StatusCode::NO_CONTENT,
        headers: Vec::new(),
        chunks: Vec::new(),
        header_delay: Duration::ZERO,
    };
    let mock = MockUpstream::start(script).await;
    let mut gw = gateway("anthropic", &mock, &[]).await;
    let resp = client()
        .delete(gw.url("/anthropic/v1/files/f1?beta=true"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);
    resp.bytes().await.unwrap();

    let r = gw.record().await;
    assert_eq!(r.outcome, Outcome::Ok);
    assert_eq!(r.http_status, Some(204));
    assert_eq!(r.method, "DELETE");
    assert_eq!(r.path, "/v1/files/f1");
}
