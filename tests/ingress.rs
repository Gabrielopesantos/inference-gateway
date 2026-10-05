#![allow(clippy::unwrap_used, clippy::expect_used)]

mod support;

use axum::http::StatusCode;
use clap::Parser;
use inference_gateway::config::Config;
use inference_gateway::config::cli::{Cli, Command};
use inference_gateway::gateway::{Gateway, Secrets};
use inference_gateway::modules::{BuildCtx, Dispatcher};

use support::mock_upstream::{MockUpstream, Script};
use support::{client, config, spawn_gateway};

async fn post(url: &str, token: Option<&str>) -> reqwest::Response {
    let mut req = client().post(url).body("{}");
    if let Some(t) = token {
        req = req.header("x-gateway-token", t);
    }
    req.send().await.unwrap()
}

#[tokio::test]
async fn client_outside_allow_cidrs_is_403() {
    let mock = MockUpstream::start(Script::json(StatusCode::OK, b"{}")).await;
    let cfg = config(
        &[("anthropic", "anthropic", &mock.base_url())],
        "[ingress]\nallow_cidrs = [\"10.0.0.0/8\"]\n",
    );
    let gw = spawn_gateway(&cfg, &[], None).await;

    let resp = post(&gw.url("/anthropic/v1/messages"), None).await;
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    assert!(mock.requests().is_empty());
}

#[tokio::test]
async fn client_inside_allow_cidrs_is_forwarded() {
    let mock = MockUpstream::start(Script::json(StatusCode::OK, b"{}")).await;
    let cfg = config(
        &[("anthropic", "anthropic", &mock.base_url())],
        "[ingress]\nallow_cidrs = [\"10.0.0.0/8\", \"127.0.0.0/8\"]\n",
    );
    let gw = spawn_gateway(&cfg, &[], None).await;

    let resp = post(&gw.url("/anthropic/v1/messages"), None).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn token_is_required_and_never_forwarded() {
    let mock = MockUpstream::start(Script::json(StatusCode::OK, b"{}")).await;
    let cfg = config(
        &[("anthropic", "anthropic", &mock.base_url())],
        "[ingress]\nrequire_token = true\n",
    );
    let gw = spawn_gateway(&cfg, &[], Some("s3cret")).await;
    let url = gw.url("/anthropic/v1/messages");

    assert_eq!(post(&url, None).await.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        post(&url, Some("wrong")).await.status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        post(&url, Some("s3cre")).await.status(),
        StatusCode::UNAUTHORIZED
    );
    assert!(mock.requests().is_empty());

    let resp = post(&url, Some("s3cret")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(!mock.last_request().headers.contains_key("x-gateway-token"));
}

#[tokio::test]
async fn require_token_without_env_fails_at_startup() {
    let cfg = Config::parse(&config(
        &[("anthropic", "anthropic", "https://api.anthropic.com")],
        "[ingress]\nrequire_token = true\n",
    ))
    .unwrap();
    let Command::Serve(args) = Cli::try_parse_from(["ig", "--config", "x.toml"])
        .unwrap()
        .into_command()
        .unwrap()
    else {
        panic!("not serve")
    };
    let dispatcher = Dispatcher::start(Vec::new(), &BuildCtx::default(), 1);
    for token in [None, Some(String::new())] {
        let secrets = Secrets {
            gateway_token: token,
            identity_secret: None,
        };
        let err = Gateway::build(cfg.clone(), &args, secrets, &dispatcher).unwrap_err();
        assert!(err.to_string().contains("IG_GATEWAY_TOKEN"), "{err}");
    }
}
