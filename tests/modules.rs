#![cfg(all(feature = "cost", feature = "prometheus"))]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use axum::http::StatusCode;
use inference_gateway::gateway::Secrets;
use inference_gateway::modules::{CompletedExchange, Module, Phase};
use tokio::time::Instant;

use support::mock_upstream::{MockUpstream, Script};
use support::{Options, TestGateway, client, config, spawn_gateway_with};

const CC_CACHE_HIT: &[u8] = include_bytes!("fixtures/anthropic/cc_cache_hit.sse");
const CC_TEXT_SSE: &[u8] = include_bytes!("fixtures/anthropic/cc_text.sse");
const CC_TEXT_HEADERS: &str = include_str!("fixtures/anthropic/cc_text.headers");
const ANTHROPIC_429: &[u8] = include_bytes!("fixtures/anthropic/rate_limited.json");
const ANTHROPIC_429_HEADERS: &str = include_str!("fixtures/anthropic/rate_limited.headers");
const OPENAI_CACHE_HIT: &[u8] = include_bytes!("fixtures/openai/cache_hit.sse");

const MESSAGES: &str = r#"{"model":"claude-sonnet-5","max_tokens":64,"stream":true,"messages":[]}"#;
const CHAT: &str = r#"{"model":"gpt-5-mini","stream":true,"messages":[]}"#;

fn modules_config(provider: &str, mock: &MockUpstream) -> String {
    config(
        &[(provider, provider, &mock.base_url())],
        &format!(
            "[cost]\npricing_file = \"{}/config/pricing.toml\"\n\n[prometheus]\n",
            env!("CARGO_MANIFEST_DIR")
        ),
    )
}

async fn gateway(provider: &str, mock: &MockUpstream, opts: Options<'_>) -> TestGateway {
    spawn_gateway_with(&modules_config(provider, mock), opts).await
}

async fn post(gw: &TestGateway, path: &str, body: &str) -> reqwest::Response {
    client()
        .post(gw.url(path))
        .header("x-api-key", "sk-test")
        .header("x-gateway-session", "run-7")
        .header("x-gateway-tag-project", "ig")
        .body(body.to_owned())
        .send()
        .await
        .unwrap()
}

fn assert_close(actual: Option<f64>, expected: f64) {
    let actual = actual.expect("cost_usd set");
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

/// Waits for the prometheus worker, which runs behind a queue.
async fn metrics_containing(gw: &TestGateway, needle: &str) -> String {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let text = gw.metrics().await;
        if text.contains(needle) || Instant::now() > deadline {
            return text;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn identity_from_headers_and_secret() {
    let mock = MockUpstream::start(Script::sse(CC_TEXT_SSE, Duration::ZERO)).await;
    let opts = Options {
        secrets: Secrets {
            identity_secret: Some("k1".into()),
            ..Secrets::default()
        },
        ..Options::default()
    };
    let mut gw = gateway("anthropic", &mock, opts).await;
    post(&gw, "/anthropic/v1/messages", MESSAGES)
        .await
        .bytes()
        .await
        .unwrap();

    let r = gw.record().await;
    // python3 -c "import hmac,hashlib;print(hmac.new(b'k1',b'sk-test',hashlib.sha256).hexdigest()[:16])"
    assert_eq!(r.principal.caller_id.as_deref(), Some("70bacccca01b7aef"));
    assert_eq!(r.principal.session_id.as_deref(), Some("run-7"));
    assert_eq!(
        r.principal.tags.get("project").map(String::as_str),
        Some("ig")
    );
    let upstream = mock.last_request().headers;
    assert!(
        !upstream
            .keys()
            .any(|k| k.as_str().starts_with("x-gateway-"))
    );
    assert_eq!(upstream["x-api-key"], "sk-test");
}

#[tokio::test]
async fn claude_code_session_header_is_read_and_forwarded() {
    let mock = MockUpstream::start(Script::sse(CC_TEXT_SSE, Duration::ZERO)).await;
    let mut gw = gateway("anthropic", &mock, Options::default()).await;
    client()
        .post(gw.url("/anthropic/v1/messages"))
        .header("x-claude-code-session-id", "cc-session-1")
        .body(MESSAGES)
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();

    let r = gw.record().await;
    assert_eq!(r.principal.session_id.as_deref(), Some("cc-session-1"));
    assert_eq!(
        mock.last_request().headers["x-claude-code-session-id"],
        "cc-session-1"
    );
}

#[tokio::test]
async fn anthropic_cache_hit_uses_cache_prices() {
    let mock = MockUpstream::start(Script::sse(CC_CACHE_HIT, Duration::ZERO)).await;
    let mut gw = gateway("anthropic", &mock, Options::default()).await;
    post(&gw, "/anthropic/v1/messages", MESSAGES)
        .await
        .bytes()
        .await
        .unwrap();

    let r = gw.record().await;
    assert_eq!(r.usage.input_tokens_total, Some(2 + 221 + 92_467));
    // claude-sonnet-5: $2 input, $0.20 cache read, $4 1h cache write, $10 output.
    let expected = (2.0 * 2.0 + 92_467.0 * 0.2 + 221.0 * 4.0 + 6.0 * 10.0) / 1e6;
    assert_close(r.cost_usd, expected);

    let text = metrics_containing(&gw, "gateway_cost_usd_total").await;
    assert!(
        text.contains(r#"gateway_tokens_total{kind="cache_read",model="claude-sonnet-5",provider="anthropic",upstream="anthropic"} 92467"#),
        "{text}"
    );
}

#[tokio::test]
async fn openai_cache_hit_uses_cache_prices() {
    let mock = MockUpstream::start(Script::sse(OPENAI_CACHE_HIT, Duration::ZERO)).await;
    let mut gw = gateway("openai", &mock, Options::default()).await;
    post(&gw, "/openai/v1/chat/completions", CHAT)
        .await
        .bytes()
        .await
        .unwrap();

    let r = gw.record().await;
    assert_eq!(r.response_model.as_deref(), Some("gpt-5-mini-2025-08-07"));
    assert_eq!(r.usage.input_tokens_total, Some(7040));
    // gpt-5-mini: $0.25 input, $0.025 cached input, $2 output.
    let expected = (128.0 * 0.25 + 6912.0 * 0.025 + 80.0 * 2.0) / 1e6;
    assert_close(r.cost_usd, expected);
}

#[tokio::test]
async fn rate_limit_headers_become_gauges() {
    let script = Script::json(StatusCode::TOO_MANY_REQUESTS, ANTHROPIC_429)
        .with_fixture_headers(ANTHROPIC_429_HEADERS);
    let mock = MockUpstream::start(script).await;
    let mut gw = gateway("anthropic", &mock, Options::default()).await;
    post(&gw, "/anthropic/v1/messages", MESSAGES)
        .await
        .bytes()
        .await
        .unwrap();
    gw.record().await;

    let text = metrics_containing(&gw, "gateway_upstream_ratelimit_remaining").await;
    for line in [
        r#"gateway_upstream_ratelimit_remaining{kind="requests",model="claude-sonnet-5",upstream="anthropic"} 0"#,
        r#"gateway_upstream_ratelimit_remaining{kind="tokens",model="claude-sonnet-5",upstream="anthropic"} 81000"#,
        r#"gateway_requests_total{model="claude-sonnet-5",outcome="upstream_error",provider="anthropic",streaming="false",upstream="anthropic"} 1"#,
    ] {
        assert!(text.contains(line), "missing {line}\n{text}");
    }
}

#[tokio::test]
async fn subscription_windows_become_gauges() {
    let script = Script::sse(CC_TEXT_SSE, Duration::ZERO).with_fixture_headers(CC_TEXT_HEADERS);
    let mock = MockUpstream::start(script).await;
    let mut gw = gateway("anthropic", &mock, Options::default()).await;
    post(&gw, "/anthropic/v1/messages", MESSAGES)
        .await
        .bytes()
        .await
        .unwrap();
    gw.record().await;

    let text = metrics_containing(&gw, "gateway_subscription_window_utilization").await;
    for line in [
        r#"gateway_subscription_window_utilization{upstream="anthropic",window="5h"} 0.13"#,
        r#"gateway_subscription_window_utilization{upstream="anthropic",window="7d"} 0.07"#,
    ] {
        assert!(text.contains(line), "missing {line}\n{text}");
    }
}

struct Sleepy;

#[async_trait]
impl Module for Sleepy {
    fn name(&self) -> &'static str {
        "sleepy"
    }

    fn phase(&self) -> Phase {
        Phase::Export
    }

    async fn export(&self, _ex: Arc<CompletedExchange>) {
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

#[tokio::test]
async fn slow_export_drops_instead_of_blocking() {
    let mock = MockUpstream::start(Script::sse(CC_TEXT_SSE, Duration::ZERO)).await;
    let opts = Options {
        modules: vec![Box::new(Sleepy)],
        capacity: Some(1),
        ..Options::default()
    };
    let gw = gateway("anthropic", &mock, opts).await;

    let started = Instant::now();
    for _ in 0..10 {
        post(&gw, "/anthropic/v1/messages", MESSAGES)
            .await
            .bytes()
            .await
            .unwrap();
    }
    // Ten requests behind a 1 s export would take 10 s if exports blocked.
    assert!(
        started.elapsed() < Duration::from_millis(900),
        "{:?}",
        started.elapsed()
    );

    let dropped: u64 = gw
        .ctx
        .drops
        .snapshot()
        .into_iter()
        .find(|(m, _)| *m == "sleepy")
        .unwrap()
        .1;
    assert!(dropped >= 8, "dropped {dropped}");
    let text = gw.metrics().await;
    assert!(
        text.contains(&format!(
            r#"gateway_module_dropped_total{{module="sleepy"}} {dropped}"#
        )),
        "{text}"
    );
}

const PRICE_LIST: &[u8] = include_bytes!("fixtures/pricing/litellm.json");

/// `[cost]` with only a fetched list, served by `prices`.
fn fetched_cost_config(mock: &MockUpstream, prices: &MockUpstream, extra: &str) -> String {
    config(
        &[("openai", "openai", &mock.base_url())],
        &format!(
            "[cost]\nsource_url = \"{}/list.json\"\n{extra}",
            prices.base_url()
        ),
    )
}

/// Sends chat requests until one is priced; panics after 2 s.
async fn priced_request(gw: &mut TestGateway) -> f64 {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        post(gw, "/openai/v1/chat/completions", CHAT)
            .await
            .bytes()
            .await
            .unwrap();
        if let Some(usd) = gw.record().await.cost_usd {
            return usd;
        }
        assert!(Instant::now() < deadline, "no price within 2 s");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

// gpt-5-mini in the list: $0.25 input, $0.025 cached input, $2 output.
const LISTED_COST: f64 = (128.0 * 0.25 + 6912.0 * 0.025 + 80.0 * 2.0) / 1e6;

#[tokio::test]
async fn fetched_price_list_prices_requests_and_survives_a_failed_refresh() {
    let mock = MockUpstream::start(Script::sse(OPENAI_CACHE_HIT, Duration::ZERO)).await;
    let prices = MockUpstream::start_seq(vec![
        Script::json(StatusCode::OK, PRICE_LIST),
        Script::json(StatusCode::INTERNAL_SERVER_ERROR, b"{}"),
    ])
    .await;
    let cfg = fetched_cost_config(&mock, &prices, "refresh = \"100ms\"\n");
    let mut gw = spawn_gateway_with(&cfg, Options::default()).await;

    assert_close(Some(priced_request(&mut gw).await), LISTED_COST);
    assert_eq!(prices.last_request().uri.path(), "/list.json");

    // Refreshes now fail; the fetched prices stay.
    let deadline = Instant::now() + Duration::from_secs(2);
    while prices.requests().len() < 3 {
        assert!(Instant::now() < deadline, "no refresh attempts");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_close(Some(priced_request(&mut gw).await), LISTED_COST);
}

#[tokio::test]
async fn price_cache_is_written_and_used_when_the_source_is_down() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("prices.json");
    let cache_line = format!("cache_file = \"{}\"\n", cache.display());
    let mock = MockUpstream::start(Script::sse(OPENAI_CACHE_HIT, Duration::ZERO)).await;

    let prices = MockUpstream::start(Script::json(StatusCode::OK, PRICE_LIST)).await;
    let mut gw = spawn_gateway_with(
        &fetched_cost_config(&mock, &prices, &cache_line),
        Options::default(),
    )
    .await;
    priced_request(&mut gw).await;
    let deadline = Instant::now() + Duration::from_secs(2);
    while !cache.exists() {
        assert!(Instant::now() < deadline, "cache not written");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    gw.shutdown(Duration::from_secs(2)).await.unwrap();
    assert_eq!(std::fs::read(&cache).unwrap(), PRICE_LIST);

    let down = MockUpstream::start(Script::json(StatusCode::NOT_FOUND, b"gone")).await;
    let mut gw = spawn_gateway_with(
        &fetched_cost_config(&mock, &down, &cache_line),
        Options::default(),
    )
    .await;
    // Priced on the first request: the cache loads before serving.
    post(&gw, "/openai/v1/chat/completions", CHAT)
        .await
        .bytes()
        .await
        .unwrap();
    assert_close(gw.record().await.cost_usd, LISTED_COST);
    assert_eq!(std::fs::read(&cache).unwrap(), PRICE_LIST);
}
