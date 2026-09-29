//! Provider adapters: classify requests and parse a copy of the response into
//! [`StreamEvent`]s. Adapters never change what the client receives.

pub mod anthropic;
pub mod openai;

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use axum::http::{HeaderMap, Method, header};
use bytes::Bytes;
use serde::Deserialize;

use crate::config::Upstream;

/// Wire format of an upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    Anthropic,
    Openai,
}

impl ProviderKind {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::Openai => "openai",
        }
    }
}

impl fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An observed endpoint. Other paths pass through unparsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Messages,
    ChatCompletions,
}

impl Operation {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Messages => "messages",
            Self::ChatCompletions => "chat_completions",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RequestInfo {
    pub model: Option<String>,
    pub stream: bool,
    pub max_tokens: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaKind {
    Text,
    Thinking,
    ToolInput,
}

/// Normalized usage; `None` = not reported yet. `input_tokens_total` includes
/// cache reads and writes for every provider.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens_total: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cache_read_tokens: Option<u64>,
    pub cache_write_tokens: Option<u64>,
    /// Part of `cache_write_tokens` written with the 1-hour TTL.
    pub cache_write_1h_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
}

impl Usage {
    /// Reported fields overwrite: providers send cumulative counts.
    pub fn merge(&mut self, patch: &Usage) {
        let fields = [
            (&mut self.input_tokens_total, patch.input_tokens_total),
            (&mut self.output_tokens, patch.output_tokens),
            (&mut self.cache_read_tokens, patch.cache_read_tokens),
            (&mut self.cache_write_tokens, patch.cache_write_tokens),
            (&mut self.cache_write_1h_tokens, patch.cache_write_1h_tokens),
            (&mut self.reasoning_tokens, patch.reasoning_tokens),
        ];
        for (field, value) in fields {
            if value.is_some() {
                *field = value;
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    ResponseModel(String),
    /// The first one sets TTFT.
    ContentDelta(DeltaKind),
    Usage(Usage),
    /// `stop_reason` / `finish_reason`.
    Stop(String),
    Error {
        kind: String,
        message: String,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RateLimitSnapshot {
    pub requests_remaining: Option<u64>,
    pub tokens_remaining: Option<u64>,
    pub input_tokens_remaining: Option<u64>,
    pub output_tokens_remaining: Option<u64>,
    pub retry_after: Option<Duration>,
    /// Subscription (Claude Code OAuth) windows, 0.0..=1.0 used.
    pub window_5h_utilization: Option<f64>,
    pub window_7d_utilization: Option<f64>,
}

pub trait StreamParser: Send {
    fn push(&mut self, chunk: &[u8], out: &mut Vec<StreamEvent>);
    fn finish(&mut self, out: &mut Vec<StreamEvent>);
}

pub trait ProviderAdapter: Send + Sync + fmt::Debug {
    fn kind(&self) -> ProviderKind;
    fn classify(&self, method: &Method, path: &str) -> Option<Operation>;
    fn inspect_request(&self, op: Operation, body: &[u8]) -> RequestInfo;
    fn rewrite_request(&self, _info: &RequestInfo, body: Bytes) -> Bytes {
        body
    }
    fn stream_parser(&self) -> Box<dyn StreamParser>;
    /// Non-stream JSON body, as the events a stream would have produced.
    fn parse_response(&self, body: &[u8]) -> Vec<StreamEvent>;
    fn rate_limits(&self, headers: &HeaderMap) -> RateLimitSnapshot;

    fn api_key<'a>(&self, headers: &'a HeaderMap) -> Option<&'a str> {
        if let Some(key) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
            return Some(key);
        }
        headers
            .get(header::AUTHORIZATION)?
            .to_str()
            .ok()?
            .strip_prefix("Bearer ")
    }
}

#[must_use]
pub fn adapter(upstream: &Upstream) -> Arc<dyn ProviderAdapter> {
    match upstream.provider {
        ProviderKind::Anthropic => Arc::new(anthropic::Anthropic),
        ProviderKind::Openai => Arc::new(openai::OpenAi {
            inject_stream_usage: upstream.inject_stream_usage,
        }),
    }
}

/// Fields shared by both providers' request bodies.
#[derive(Deserialize)]
struct RequestFields {
    model: Option<String>,
    #[serde(default)]
    stream: bool,
    max_tokens: Option<u64>,
    max_completion_tokens: Option<u64>,
}

fn inspect_common(body: &[u8]) -> RequestInfo {
    match serde_json::from_slice::<RequestFields>(body) {
        Ok(f) => RequestInfo {
            model: f.model,
            stream: f.stream,
            max_tokens: f.max_completion_tokens.or(f.max_tokens),
        },
        Err(_) => RequestInfo::default(),
    }
}

fn header_u64(headers: &HeaderMap, name: &str) -> Option<u64> {
    headers.get(name)?.to_str().ok()?.trim().parse().ok()
}

fn header_f64(headers: &HeaderMap, name: &str) -> Option<f64> {
    headers
        .get(name)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()
        .filter(|v: &f64| v.is_finite())
}

/// Delta-seconds form only; the HTTP-date form is ignored.
fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get(header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|s| s.is_finite() && *s >= 0.0)
        .map(Duration::from_secs_f64)
}

/// Shared test helpers: parse fixtures in one piece and in random splits.
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
pub(crate) mod testing {
    use super::{DeltaKind, ProviderAdapter, StreamEvent, Usage};

    pub fn total_usage(events: &[StreamEvent]) -> Usage {
        let mut u = Usage::default();
        for e in events {
            if let StreamEvent::Usage(p) = e {
                u.merge(p);
            }
        }
        u
    }

    pub fn deltas(events: &[StreamEvent]) -> Vec<DeltaKind> {
        let mut out: Vec<DeltaKind> = Vec::new();
        for e in events {
            if let StreamEvent::ContentDelta(k) = e
                && out.last() != Some(k)
            {
                out.push(*k);
            }
        }
        out
    }

    pub fn parse_all(adapter: &dyn ProviderAdapter, chunks: &[&[u8]]) -> Vec<StreamEvent> {
        let mut p = adapter.stream_parser();
        let mut out = Vec::new();
        for c in chunks {
            p.push(c, &mut out);
        }
        p.finish(&mut out);
        out
    }

    pub fn split_at(text: &[u8], cuts: &[usize]) -> Vec<Vec<u8>> {
        let mut cuts: Vec<usize> = cuts.iter().map(|c| c % (text.len() + 1)).collect();
        cuts.sort_unstable();
        let mut out = Vec::new();
        let mut prev = 0;
        for c in cuts {
            out.push(text[prev..c].to_vec());
            prev = c;
        }
        out.push(text[prev..].to_vec());
        out
    }

    /// Every `*.sse` fixture of a provider, so newly captured ones are covered.
    pub fn sse_fixtures(provider: &str) -> Vec<(String, Vec<u8>)> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(provider);
        let mut out: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "sse"))
            .map(|p| (p.display().to_string(), std::fs::read(&p).unwrap()))
            .collect();
        out.sort();
        out
    }

    pub fn headers(raw: &str) -> axum::http::HeaderMap {
        let mut map = axum::http::HeaderMap::new();
        for line in raw.lines().skip(1) {
            if let Some((k, v)) = line.split_once(':') {
                map.append(
                    axum::http::HeaderName::from_bytes(k.trim().as_bytes()).unwrap(),
                    v.trim().parse().unwrap(),
                );
            }
        }
        map
    }
}
