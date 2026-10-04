#![cfg(feature = "store")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::time::Duration;

use support::mock_upstream::{MockUpstream, Script};
use support::{TestGateway, client, config, spawn_gateway};

const TEXT_SSE: &[u8] = include_bytes!("fixtures/anthropic/text.sse");
const OPENAI_SSE: &[u8] = include_bytes!("fixtures/openai/text.sse");

const MESSAGES: &str = r#"{"model":"claude-sonnet-5","max_tokens":64,"stream":true,"messages":[]}"#;
const CHAT: &str = r#"{"model":"gpt-5-mini","stream":true,"messages":[]}"#;

async fn gateway(
    provider: &str,
    mock: &MockUpstream,
    rules: &str,
) -> (TestGateway, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let extra = format!(
        "[store]\npath = \"{}\"\n\n[capture]\nmax_bytes = \"1 MiB\"\n\n{rules}",
        dir.path().join("gateway.db").display()
    );
    let gw = spawn_gateway(
        &config(&[(provider, provider, &mock.base_url())], &extra),
        &[],
        None,
    )
    .await;
    (gw, dir)
}

async fn post(gw: &TestGateway, path: &str, body: &str, capture: Option<&str>) {
    let mut req = client().post(gw.url(path)).body(body.to_owned());
    if let Some(v) = capture {
        req = req.header("x-gateway-capture", v);
    }
    req.send().await.unwrap().bytes().await.unwrap();
}

#[tokio::test]
async fn header_overrides_rule_when_allowed() {
    let mock = MockUpstream::start(Script::sse(TEXT_SSE, Duration::ZERO)).await;
    let rules = "[[capture.rules]]\ncapture = false\nallow_header_override = true\n";
    let (mut gw, _dir) = gateway("anthropic", &mock, rules).await;

    post(&gw, "/anthropic/v1/messages", MESSAGES, Some("1")).await;
    let ex = gw.exchange().await;
    let c = ex.content.unwrap();
    assert_eq!(&c.request_body[..], MESSAGES.as_bytes());
    assert_eq!(&c.raw_response[..], TEXT_SSE);
    assert!(
        !mock
            .last_request()
            .headers
            .contains_key("x-gateway-capture")
    );

    post(&gw, "/anthropic/v1/messages", MESSAGES, None).await;
    let ex = gw.exchange().await;
    assert!(ex.content.is_none());
}

#[tokio::test]
async fn header_ignored_when_not_allowed() {
    let mock = MockUpstream::start(Script::sse(TEXT_SSE, Duration::ZERO)).await;
    let rules = "[[capture.rules]]\nmatch = { model = \"claude-*\" }\n";
    let (mut gw, _dir) = gateway("anthropic", &mock, rules).await;

    post(&gw, "/anthropic/v1/messages", MESSAGES, Some("0")).await;
    assert!(gw.exchange().await.content.is_some());
}

#[tokio::test]
async fn unobserved_paths_are_never_captured() {
    let mock = MockUpstream::start(Script::sse(TEXT_SSE, Duration::ZERO)).await;
    let (mut gw, _dir) = gateway("anthropic", &mock, "[[capture.rules]]\n").await;

    client()
        .get(gw.url("/anthropic/v1/models"))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert!(gw.exchange().await.content.is_none());
}

#[tokio::test]
async fn stores_client_body_not_rewritten_body() {
    let mock = MockUpstream::start(Script::sse(OPENAI_SSE, Duration::ZERO)).await;
    let (mut gw, _dir) = gateway("openai", &mock, "[[capture.rules]]\n").await;

    post(&gw, "/openai/v1/chat/completions", CHAT, None).await;
    let c = gw.exchange().await.content.unwrap();
    assert_eq!(&c.request_body[..], CHAT.as_bytes());
    let sent: serde_json::Value = serde_json::from_slice(&mock.last_request().body).unwrap();
    assert_eq!(sent["stream_options"]["include_usage"], true);
}
