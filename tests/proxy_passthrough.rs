#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use std::time::Duration;

use axum::http::{HeaderMap, StatusCode};
use tokio::time::Instant;

use support::mock_upstream::{MockUpstream, Script, unreachable_addr};
use support::{client, config, spawn_gateway};

const ANTHROPIC_SSE: &[u8] = include_bytes!("fixtures/anthropic/text.sse");
const OPENAI_SSE: &[u8] = include_bytes!("fixtures/openai/text.sse");
const ANTHROPIC_JSON: &[u8] = include_bytes!("fixtures/anthropic/text.json");

/// Headers that legitimately differ between a direct and a proxied response.
const EXCLUDED: &[&str] = &["connection", "keep-alive", "transfer-encoding", "date"];

fn comparable(headers: &HeaderMap) -> Vec<(String, String)> {
    let mut v: Vec<_> = headers
        .iter()
        .filter(|(n, _)| !EXCLUDED.contains(&n.as_str()))
        .map(|(n, v)| (n.to_string(), v.to_str().unwrap().to_owned()))
        .collect();
    v.sort();
    v
}

async fn assert_passthrough_identical(provider: &str, path: &str, script: Script) {
    let expected_body = script.body();
    let mock = MockUpstream::start(script).await;
    let gw = spawn_gateway(
        &config(&[(provider, provider, &mock.base_url())], ""),
        &[],
        None,
    )
    .await;
    let http = client();
    // Already has `include_usage`, so the openai adapter leaves it unchanged.
    let req_body =
        r#"{"model":"m","stream":true,"stream_options":{"include_usage":true},"messages":[]}"#;

    let direct = http
        .post(format!("{}{path}", mock.base_url()))
        .body(req_body)
        .send()
        .await
        .unwrap();
    let direct_status = direct.status();
    let direct_headers = direct.headers().clone();
    let direct_body = direct.bytes().await.unwrap();

    let proxied = http
        .post(gw.url(&format!("/{provider}{path}")))
        .body(req_body)
        .send()
        .await
        .unwrap();
    assert_eq!(proxied.status(), direct_status);
    assert_eq!(comparable(proxied.headers()), comparable(&direct_headers));
    let proxied_body = proxied.bytes().await.unwrap();

    assert_eq!(direct_body.as_ref(), expected_body.as_slice());
    assert_eq!(proxied_body.as_ref(), expected_body.as_slice());
    assert_eq!(mock.last_request().body.as_ref(), req_body.as_bytes());
}

#[tokio::test]
async fn anthropic_stream_is_byte_identical() {
    let script = Script::sse(ANTHROPIC_SSE, Duration::from_millis(5));
    assert_passthrough_identical("anthropic", "/v1/messages", script).await;
}

#[tokio::test]
async fn openai_stream_is_byte_identical() {
    let script = Script::sse(OPENAI_SSE, Duration::from_millis(5));
    assert_passthrough_identical("openai", "/v1/chat/completions", script).await;
}

#[tokio::test]
async fn json_response_keeps_content_length() {
    let script = Script::json(StatusCode::OK, ANTHROPIC_JSON);
    assert_passthrough_identical("anthropic", "/v1/messages", script).await;
}

#[tokio::test]
async fn upstream_error_status_passes_through() {
    let body = br#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#;
    let mut script = Script::json(StatusCode::TOO_MANY_REQUESTS, body);
    script.headers.push(("retry-after".into(), "7".into()));
    let mock = MockUpstream::start(script).await;
    let gw = spawn_gateway(
        &config(&[("anthropic", "anthropic", &mock.base_url())], ""),
        &[],
        None,
    )
    .await;

    let resp = client()
        .post(gw.url("/anthropic/v1/messages"))
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(resp.headers()["retry-after"], "7");
    assert_eq!(resp.bytes().await.unwrap().as_ref(), body);
}

#[tokio::test]
async fn gateway_headers_stripped_keys_forwarded() {
    let mock = MockUpstream::start(Script::json(StatusCode::OK, b"{}")).await;
    let gw = spawn_gateway(
        &config(&[("anthropic", "anthropic", &mock.base_url())], ""),
        &[],
        None,
    )
    .await;

    client()
        .post(gw.url("/anthropic/v1/messages"))
        .header("x-api-key", "sk-ant-test-key")
        .header("authorization", "Bearer sk-test")
        .header("anthropic-version", "2023-06-01")
        .header("anthropic-beta", "prompt-caching-2024-07-31")
        .header("x-gateway-session", "run-1")
        .header("x-gateway-tag-project", "gw")
        .header("x-gateway-capture", "1")
        .header("accept-encoding", "gzip, br")
        .body("{}")
        .send()
        .await
        .unwrap();

    let seen = mock.last_request().headers;
    assert_eq!(seen["x-api-key"], "sk-ant-test-key");
    assert_eq!(seen["authorization"], "Bearer sk-test");
    assert_eq!(seen["anthropic-version"], "2023-06-01");
    assert_eq!(seen["anthropic-beta"], "prompt-caching-2024-07-31");
    assert_eq!(seen["accept-encoding"], "identity");
    assert!(
        seen.keys().all(|n| !n.as_str().starts_with("x-gateway-")),
        "{seen:?}"
    );
}

#[tokio::test]
async fn path_and_query_preserved() {
    let mock = MockUpstream::start(Script::json(StatusCode::OK, b"{}")).await;
    let gw = spawn_gateway(
        &config(&[("openai", "openai", &mock.base_url())], ""),
        &[],
        None,
    )
    .await;

    let resp = client()
        .get(gw.url("/openai/v1/models?limit=2&after=a%20b"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let seen = mock.last_request();
    assert_eq!(seen.method, "GET");
    assert_eq!(seen.uri, "/v1/models?limit=2&after=a%20b");
}

#[tokio::test]
async fn base_url_with_path_prefix() {
    let mock = MockUpstream::start(Script::json(StatusCode::OK, b"{}")).await;
    let base = format!("{}/api", mock.base_url());
    let gw = spawn_gateway(&config(&[("openrouter", "openai", &base)], ""), &[], None).await;

    client()
        .post(gw.url("/openrouter/v1/chat/completions"))
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(mock.last_request().uri, "/api/v1/chat/completions");
}

#[tokio::test]
async fn chunks_are_not_buffered() {
    let script = Script::chunks(vec![
        (Duration::ZERO, "data: one\n\n"),
        (Duration::from_millis(500), "data: two\n\n"),
    ]);
    let mock = MockUpstream::start(script).await;
    let gw = spawn_gateway(
        &config(&[("openai", "openai", &mock.base_url())], ""),
        &[],
        None,
    )
    .await;

    let mut resp = client()
        .post(gw.url("/openai/v1/chat/completions"))
        .body("{}")
        .send()
        .await
        .unwrap();
    let first = resp.chunk().await.unwrap().unwrap();
    let first_at = Instant::now();
    assert_eq!(first.as_ref(), b"data: one\n\n");

    let remaining = resp.bytes().await.unwrap();
    assert_eq!(remaining.as_ref(), b"data: two\n\n");
    let sent = mock.sent_at();
    assert!(
        first_at < sent[1],
        "chunk 1 arrived only after chunk 2 was sent"
    );
}

#[tokio::test]
async fn client_disconnect_cancels_upstream() {
    let script = Script::chunks(vec![
        (Duration::ZERO, "data: one\n\n"),
        (Duration::from_secs(10), "data: two\n\n"),
    ]);
    let mock = MockUpstream::start(script).await;
    let gw = spawn_gateway(
        &config(&[("anthropic", "anthropic", &mock.base_url())], ""),
        &[],
        None,
    )
    .await;

    let mut resp = client()
        .post(gw.url("/anthropic/v1/messages"))
        .body("{}")
        .send()
        .await
        .unwrap();
    resp.chunk().await.unwrap().unwrap();
    drop(resp);
    let dropped_at = Instant::now();

    let seen = mock
        .wait_disconnect(Duration::from_secs(2))
        .await
        .expect("upstream never saw the disconnect");
    let lag = seen.saturating_duration_since(dropped_at);
    assert!(
        lag < Duration::from_millis(100),
        "upstream saw disconnect after {lag:?}"
    );
}

#[tokio::test]
async fn oversized_body_is_413() {
    let mock = MockUpstream::start(Script::json(StatusCode::OK, b"{}")).await;
    let gw = spawn_gateway(
        &config(&[("anthropic", "anthropic", &mock.base_url())], ""),
        &["--max-body", "1KiB"],
        None,
    )
    .await;
    let http = client();

    // Declared length over the limit.
    let resp = http
        .post(gw.url("/anthropic/v1/messages"))
        .body(vec![b'x'; 2048])
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);

    // Chunked body with no declared length.
    let chunks: Vec<Result<Vec<u8>, std::io::Error>> =
        vec![Ok(vec![b'x'; 800]), Ok(vec![b'x'; 800])];
    let resp = http
        .post(gw.url("/anthropic/v1/messages"))
        .body(reqwest::Body::wrap_stream(futures_util::stream::iter(
            chunks,
        )))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let v: serde_json::Value = serde_json::from_slice(&resp.bytes().await.unwrap()).unwrap();
    assert_eq!(v["error"]["type"], "gateway_body_too_large");

    // Exactly at the limit passes.
    let resp = http
        .post(gw.url("/anthropic/v1/messages"))
        .body(vec![b'x'; 1024])
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(mock.requests().len(), 1);
}

#[tokio::test]
async fn unknown_upstream_is_404() {
    let gw = spawn_gateway(
        &config(&[("anthropic", "anthropic", "http://127.0.0.1:9")], ""),
        &[],
        None,
    )
    .await;
    for path in ["/nope/v1/messages", "/", "/nope"] {
        let resp = client().post(gw.url(path)).body("{}").send().await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND, "{path}");
        let v: serde_json::Value = serde_json::from_slice(&resp.bytes().await.unwrap()).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "gateway_unknown_upstream");
    }
}

#[tokio::test]
async fn unreachable_upstream_is_502() {
    let base = format!("http://{}", unreachable_addr());
    let gw = spawn_gateway(&config(&[("openai", "openai", &base)], ""), &[], None).await;
    let resp = client()
        .post(gw.url("/openai/v1/chat/completions"))
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    let v: serde_json::Value = serde_json::from_slice(&resp.bytes().await.unwrap()).unwrap();
    assert_eq!(v["error"]["type"], "gateway_upstream_unreachable");
}

#[tokio::test]
async fn silent_upstream_before_headers_is_504() {
    let mut script = Script::json(StatusCode::OK, b"{}");
    script.header_delay = Duration::from_secs(2);
    let mock = MockUpstream::start(script).await;
    let gw = spawn_gateway(
        &config(&[("anthropic", "anthropic", &mock.base_url())], ""),
        &["--idle-timeout", "200ms"],
        None,
    )
    .await;
    let resp = client()
        .post(gw.url("/anthropic/v1/messages"))
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::GATEWAY_TIMEOUT);
}

#[tokio::test]
async fn silent_upstream_mid_stream_aborts_client() {
    let script = Script::chunks(vec![
        (Duration::ZERO, "data: one\n\n"),
        (Duration::from_secs(2), "data: two\n\n"),
    ]);
    let mock = MockUpstream::start(script).await;
    let gw = spawn_gateway(
        &config(&[("anthropic", "anthropic", &mock.base_url())], ""),
        &["--idle-timeout", "200ms"],
        None,
    )
    .await;
    let mut resp = client()
        .post(gw.url("/anthropic/v1/messages"))
        .body("{}")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    resp.chunk().await.unwrap().unwrap();
    // The client must see an error, not a clean end of a truncated stream.
    assert!(resp.chunk().await.is_err());
}

#[tokio::test]
async fn admin_endpoints() {
    let gw = spawn_gateway(
        &config(&[("anthropic", "anthropic", "http://127.0.0.1:9")], ""),
        &[],
        None,
    )
    .await;
    let http = client();
    let resp = http.get(gw.admin_url("/healthz")).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.text().await.unwrap(), "ok");
    // Served only with [prometheus].
    let resp = http.get(gw.admin_url("/metrics")).send().await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn graceful_shutdown_drains_in_flight_stream() {
    let script = Script::chunks(vec![
        (Duration::ZERO, "data: one\n\n"),
        (Duration::from_millis(300), "data: two\n\n"),
    ]);
    let mock = MockUpstream::start(script).await;
    let gw = spawn_gateway(
        &config(&[("anthropic", "anthropic", &mock.base_url())], ""),
        &[],
        None,
    )
    .await;
    let addr = gw.addr;

    let mut resp = client()
        .post(gw.url("/anthropic/v1/messages"))
        .body("{}")
        .send()
        .await
        .unwrap();
    resp.chunk().await.unwrap().unwrap();

    let shutdown = tokio::spawn(gw.shutdown(Duration::from_secs(5)));
    let remaining = resp.bytes().await.unwrap();
    assert_eq!(remaining.as_ref(), b"data: two\n\n");
    shutdown.await.unwrap().unwrap();

    // Stopped accepting.
    assert!(tokio::net::TcpStream::connect(addr).await.is_err());
}
