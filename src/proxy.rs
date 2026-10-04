//! Passthrough forwarding: `/{upstream}/{rest}?q` goes to `base_url/rest?q`.
//! The request body is buffered; the response body streams back unbuffered.
//!
//! Client disconnect cancels the upstream request for free: dropping the
//! response body drops the upstream stream and its connection.

use std::collections::HashSet;
use std::error::Error as _;
use std::sync::Arc;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::header::{self, HeaderMap, HeaderName, HeaderValue};
use axum::http::request::Parts;
use axum::response::Response;
use bytes::Bytes;
use futures_util::TryStreamExt;
use http_body_util::{BodyExt, LengthLimitError, Limited};
use tokio::time::Instant;
use tracing::{debug, warn};
use url::Url;

use crate::error::GatewayError;
use crate::gateway::{Gateway, Route};
use crate::identity::Principal;
use crate::observe::Observation;
use crate::redact;

/// Prefix of gateway-only request headers; never forwarded.
pub const GATEWAY_HEADER_PREFIX: &str = "x-gateway-";

/// RFC 9110 §7.6.1 plus legacy hop-by-hop headers.
const HOP_BY_HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-connection",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

/// # Errors
///
/// Unknown upstream, bad or oversized body, or upstream unreachable.
pub async fn forward(
    State(gw): State<Arc<Gateway>>,
    req: Request,
) -> Result<Response, GatewayError> {
    let start = Instant::now();
    let (parts, body) = req.into_parts();
    let (name, upstream_path) = split_path(parts.uri.path());
    let Some(route) = gw.route(name) else {
        warn!(
            upstream = name,
            path = parts.uri.path(),
            "unknown upstream; client base URL must end in /{{upstream}}"
        );
        return Err(GatewayError::UnknownUpstream(name.to_owned()));
    };
    let mut obs = Observation::new(
        &route.upstream.name,
        route.adapter.clone(),
        gw.sink.clone(),
        start,
    );
    obs.target(&parts.method, upstream_path);
    obs.principal(Principal::from_headers(
        &parts.headers,
        route.adapter.api_key(&parts.headers),
        gw.identity.as_ref(),
    ));

    let upstream_resp = match send(&gw, route, &parts, upstream_path, body, &mut obs).await {
        Ok(resp) => resp,
        Err(e) => {
            obs.fail(&e);
            return Err(e);
        }
    };

    obs.response(upstream_resp.status(), upstream_resp.headers());
    let mut out = Response::new(Body::empty());
    *out.status_mut() = upstream_resp.status();
    *out.headers_mut() = downstream_headers(upstream_resp.headers());
    let upstream_name = route.upstream.name.clone();
    let stream = upstream_resp.bytes_stream().inspect_err(move |e| {
        warn!(upstream = %upstream_name, error = %error_chain(e), "upstream body failed mid-stream");
    });
    *out.body_mut() = Body::from_stream(obs.into_body(stream));
    Ok(out)
}

async fn send(
    gw: &Gateway,
    route: &Route,
    parts: &Parts,
    rest: &str,
    body: Body,
    obs: &mut Observation,
) -> Result<reqwest::Response, GatewayError> {
    let upstream = &route.upstream;
    let url = upstream_url(&upstream.base_url, rest, parts.uri.query())?;

    if content_length(&parts.headers).is_some_and(|len| len > gw.max_body) {
        return Err(GatewayError::BodyTooLarge { limit: gw.max_body });
    }
    let mut body = read_body(body, gw.max_body).await?;

    if let Some(op) = route.adapter.classify(&parts.method, rest) {
        let info = route.adapter.inspect_request(op, &body);
        body = route.adapter.rewrite_request(&info, body);
        obs.request(op, info);
    }

    let headers = upstream_headers(&parts.headers);
    debug!(
        upstream = %upstream.name,
        method = %parts.method,
        path = rest,
        headers = ?redact::headers(&headers),
        body_bytes = body.len(),
        "forwarding request"
    );

    gw.client
        .request(parts.method.clone(), url)
        .headers(headers)
        .body(body)
        .send()
        .await
        .map_err(|e| {
            let e = e.without_url();
            warn!(upstream = %upstream.name, error = %error_chain(&e), "upstream request failed");
            if e.is_timeout() {
                GatewayError::UpstreamTimeout(error_chain(&e))
            } else {
                GatewayError::UpstreamUnreachable(error_chain(&e))
            }
        })
}

/// `rest` keeps its leading `/` and percent-encoding; empty for `/{upstream}`.
fn split_path(path: &str) -> (&str, &str) {
    let path = path.strip_prefix('/').unwrap_or(path);
    match path.find('/') {
        Some(i) => (&path[..i], &path[i..]),
        None => (path, ""),
    }
}

fn upstream_url(base: &Url, rest: &str, query: Option<&str>) -> Result<Url, GatewayError> {
    let mut url = base.as_str().trim_end_matches('/').to_owned();
    url.push_str(rest);
    if let Some(q) = query {
        url.push('?');
        url.push_str(q);
    }
    Url::parse(&url).map_err(|e| GatewayError::InvalidPath(e.to_string()))
}

fn content_length(headers: &HeaderMap) -> Option<u64> {
    headers
        .get(header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

async fn read_body(body: Body, limit: u64) -> Result<Bytes, GatewayError> {
    let max = usize::try_from(limit).unwrap_or(usize::MAX);
    match Limited::new(body, max).collect().await {
        Ok(collected) => Ok(collected.to_bytes()),
        Err(e) if e.is::<LengthLimitError>() => Err(GatewayError::BodyTooLarge { limit }),
        Err(e) => Err(GatewayError::BadRequestBody(e.to_string())),
    }
}

/// Names listed in `Connection` are hop-by-hop for this message too.
fn connection_listed(headers: &HeaderMap) -> HashSet<String> {
    headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .map(|n| n.trim().to_ascii_lowercase())
        .filter(|n| !n.is_empty())
        .collect()
}

fn is_hop_by_hop(name: &HeaderName, listed: &HashSet<String>) -> bool {
    HOP_BY_HOP.contains(&name.as_str()) || listed.contains(name.as_str())
}

/// `content-length` is recomputed from the buffered body. `identity` keeps the
/// response parseable without decompression.
fn upstream_headers(incoming: &HeaderMap) -> HeaderMap {
    let listed = connection_listed(incoming);
    let mut out = HeaderMap::with_capacity(incoming.len());
    for (name, value) in incoming {
        if is_hop_by_hop(name, &listed)
            || name == header::HOST
            || name == header::CONTENT_LENGTH
            || name == header::ACCEPT_ENCODING
            || name.as_str().starts_with(GATEWAY_HEADER_PREFIX)
        {
            continue;
        }
        out.append(name.clone(), value.clone());
    }
    out.insert(
        header::ACCEPT_ENCODING,
        HeaderValue::from_static("identity"),
    );
    out
}

fn downstream_headers(upstream: &HeaderMap) -> HeaderMap {
    let listed = connection_listed(upstream);
    let mut out = HeaderMap::with_capacity(upstream.len());
    for (name, value) in upstream {
        if !is_hop_by_hop(name, &listed) {
            out.append(name.clone(), value.clone());
        }
    }
    out
}

/// reqwest's top-level message alone ("error sending request") hides the cause.
fn error_chain(e: &reqwest::Error) -> String {
    let mut msg = e.to_string();
    let mut source = e.source();
    while let Some(s) = source {
        msg.push_str(": ");
        msg.push_str(&s.to_string());
        source = s.source();
    }
    msg
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn url(base: &str, rest: &str, query: Option<&str>) -> String {
        upstream_url(&Url::parse(base).unwrap(), rest, query)
            .unwrap()
            .to_string()
    }

    #[test]
    fn splits_upstream_and_rest() {
        assert_eq!(
            split_path("/anthropic/v1/messages"),
            ("anthropic", "/v1/messages")
        );
        assert_eq!(split_path("/anthropic/"), ("anthropic", "/"));
        assert_eq!(split_path("/anthropic"), ("anthropic", ""));
        assert_eq!(split_path("/"), ("", ""));
    }

    #[test]
    fn builds_upstream_url() {
        assert_eq!(
            url("https://api.anthropic.com", "/v1/messages", None),
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(
            url(
                "https://api.openai.com",
                "/v1/models",
                Some("limit=2&after=x")
            ),
            "https://api.openai.com/v1/models?limit=2&after=x"
        );
    }

    #[test]
    fn keeps_base_url_path() {
        assert_eq!(
            url("https://openrouter.ai/api", "/v1/chat/completions", None),
            "https://openrouter.ai/api/v1/chat/completions"
        );
        assert_eq!(
            url("https://openrouter.ai/api/", "/v1/chat/completions", None),
            "https://openrouter.ai/api/v1/chat/completions"
        );
    }

    #[test]
    fn keeps_percent_encoding() {
        assert_eq!(
            url("http://h", "/v1/files/a%2Fb", Some("q=a%20b")),
            "http://h/v1/files/a%2Fb?q=a%20b"
        );
    }

    fn map(pairs: &[(&'static str, &'static str)]) -> HeaderMap {
        let mut m = HeaderMap::new();
        for (k, v) in pairs {
            m.append(*k, HeaderValue::from_static(v));
        }
        m
    }

    #[test]
    fn upstream_headers_strip_hop_by_hop_and_gateway() {
        let out = upstream_headers(&map(&[
            ("host", "localhost:8080"),
            ("connection", "keep-alive, x-custom-hop"),
            ("x-custom-hop", "1"),
            ("keep-alive", "timeout=5"),
            ("transfer-encoding", "chunked"),
            ("content-length", "12"),
            ("accept-encoding", "gzip, br"),
            ("x-gateway-session", "s1"),
            ("x-gateway-tag-project", "p"),
            ("x-gateway-token", "t"),
            ("x-api-key", "sk-ant"),
            ("authorization", "Bearer sk"),
            ("anthropic-version", "2023-06-01"),
            ("anthropic-beta", "a"),
            ("anthropic-beta", "b"),
        ]));
        let names: Vec<_> = out.keys().map(HeaderName::as_str).collect();
        assert_eq!(
            names,
            [
                "x-api-key",
                "authorization",
                "anthropic-version",
                "anthropic-beta",
                "accept-encoding"
            ]
        );
        assert_eq!(out["x-api-key"], "sk-ant");
        assert_eq!(out["authorization"], "Bearer sk");
        assert_eq!(out.get_all("anthropic-beta").iter().count(), 2);
        assert_eq!(out["accept-encoding"], "identity");
    }

    #[test]
    fn downstream_headers_keep_content_headers() {
        let out = downstream_headers(&map(&[
            ("content-type", "text/event-stream"),
            ("content-length", "100"),
            ("transfer-encoding", "chunked"),
            ("connection", "close"),
            ("request-id", "req_1"),
            ("anthropic-ratelimit-requests-remaining", "10"),
        ]));
        let names: Vec<_> = out.keys().map(HeaderName::as_str).collect();
        assert_eq!(
            names,
            [
                "content-type",
                "content-length",
                "request-id",
                "anthropic-ratelimit-requests-remaining"
            ]
        );
    }
}
