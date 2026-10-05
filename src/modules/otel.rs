//! One span per request, sent as OTLP/HTTP JSON. Spans are built from the
//! finished record, so no tracing SDK is involved. Attributes follow the
//! `gen_ai.*` semantic conventions; no content and no headers.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::header::CONTENT_TYPE;
use serde_json::{Map, Value, json};
use tokio::task::JoinHandle;
use tracing::{info, warn};
use url::Url;

use super::{CompletedExchange, Module, Phase};
use crate::capture;
use crate::config::OtelConfig;
use crate::observe::record::{Outcome, RequestRecord};

/// Spans per request to the collector.
const MAX_BATCH: usize = 512;
const SCOPE: &str = "inference-gateway";
const SPAN_KIND_CLIENT: u8 = 3;
const STATUS_ERROR: u8 = 2;

pub struct Otel {
    shared: Arc<Shared>,
    ticker: JoinHandle<()>,
}

struct Shared {
    client: reqwest::Client,
    url: Url,
    resource: Value,
    sample_ratio: f64,
    pending: Mutex<Vec<Value>>,
    /// Logs once per failure streak, not once per batch.
    failing: AtomicBool,
}

impl Otel {
    /// Spawns the flush ticker; call inside a tokio runtime.
    ///
    /// # Errors
    ///
    /// HTTP client cannot be built.
    pub fn new(cfg: &OtelConfig) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(cfg.timeout)
            .build()
            .context("building OTLP HTTP client")?;
        let shared = Arc::new(Shared {
            client,
            url: cfg.traces_url.clone(),
            resource: json!({ "attributes": [
                attr("service.name", json!({ "stringValue": cfg.service_name })),
                attr("service.version", json!({ "stringValue": env!("CARGO_PKG_VERSION") })),
            ]}),
            sample_ratio: cfg.sample_ratio,
            pending: Mutex::new(Vec::new()),
            failing: AtomicBool::new(false),
        });
        let ticker = tokio::spawn(flush_loop(shared.clone(), cfg.flush));
        Ok(Self { shared, ticker })
    }
}

impl Drop for Otel {
    fn drop(&mut self) {
        self.ticker.abort();
    }
}

async fn flush_loop(shared: Arc<Shared>, every: Duration) {
    let mut tick = tokio::time::interval(every);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        shared.flush().await;
    }
}

impl Shared {
    fn sampled(&self, r: &RequestRecord) -> bool {
        match r.trace_parent {
            Some(p) => p.sampled,
            None => capture::sampled(&r.otel_trace_id(), self.sample_ratio),
        }
    }

    /// Takes every pending span when `full_only` is false, else only a full batch.
    fn take(&self, full_only: bool) -> Vec<Value> {
        let mut pending = self.pending.lock().unwrap_or_else(PoisonError::into_inner);
        if full_only && pending.len() < MAX_BATCH {
            return Vec::new();
        }
        let n = pending.len().min(MAX_BATCH);
        pending.drain(..n).collect()
    }

    async fn flush(&self) {
        loop {
            let spans = self.take(false);
            if spans.is_empty() {
                return;
            }
            self.send(spans).await;
        }
    }

    /// One attempt; spans of a failed batch are dropped.
    async fn send(&self, spans: Vec<Value>) {
        let n = spans.len();
        let body = json!({ "resourceSpans": [{
            "resource": self.resource,
            "scopeSpans": [{
                "scope": { "name": SCOPE, "version": env!("CARGO_PKG_VERSION") },
                "spans": spans,
            }],
        }]});
        let result = self
            .client
            .post(self.url.clone())
            .header(CONTENT_TYPE, "application/json")
            .body(body.to_string())
            .send()
            .await
            .and_then(reqwest::Response::error_for_status);
        match result {
            Ok(_) => {
                if self.failing.swap(false, Ordering::Relaxed) {
                    info!(url = %self.url, "OTLP export recovered");
                }
            }
            Err(e) => {
                if !self.failing.swap(true, Ordering::Relaxed) {
                    warn!(url = %self.url, error = %e, spans = n, "OTLP export failed; dropping spans until it recovers");
                }
            }
        }
    }
}

#[async_trait]
impl Module for Otel {
    fn name(&self) -> &'static str {
        "otel"
    }

    fn phase(&self) -> Phase {
        Phase::Export
    }

    async fn export(&self, ex: Arc<CompletedExchange>) {
        if !self.shared.sampled(&ex.record) {
            return;
        }
        self.shared
            .pending
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(span(&ex.record));
        let batch = self.shared.take(true);
        if !batch.is_empty() {
            self.shared.send(batch).await;
        }
    }

    async fn shutdown(&self) {
        self.ticker.abort();
        self.shared.flush().await;
    }
}

/// The OTLP JSON span for one record.
#[must_use]
pub fn span(r: &RequestRecord) -> Value {
    let start = r.started_at;
    let mut a = Attrs::default();
    if r.operation.is_some() {
        a.str("gen_ai.operation.name", Some("chat"));
    }
    a.str("gen_ai.provider.name", Some(r.provider.as_str()));
    a.str("gen_ai.system", Some(r.provider.as_str()));
    a.str("gen_ai.request.model", r.requested_model.as_deref());
    a.str("gen_ai.response.model", r.response_model.as_deref());
    a.int("gen_ai.usage.input_tokens", r.usage.input_tokens_total);
    a.int("gen_ai.usage.output_tokens", r.usage.output_tokens);
    a.int(
        "gen_ai.usage.cache_read.input_tokens",
        r.usage.cache_read_tokens,
    );
    a.int(
        "gen_ai.usage.cache_creation.input_tokens",
        r.usage.cache_write_tokens,
    );
    if let Some(reason) = &r.stop_reason {
        a.push(
            "gen_ai.response.finish_reasons",
            json!({ "arrayValue": { "values": [{ "stringValue": reason }] } }),
        );
    }
    a.str("http.request.method", Some(&r.method));
    a.str("url.path", Some(&r.path));
    a.int("http.response.status_code", r.http_status.map(u64::from));
    a.str("error.type", r.error_type.as_deref());
    a.str("gateway.trace_id", Some(&r.id.simple().to_string()));
    a.str("gateway.upstream", Some(&r.upstream));
    a.str("gateway.outcome", Some(r.outcome.as_str()));
    a.push("gateway.streaming", json!({ "boolValue": r.streaming }));
    a.f64("gateway.ttfb_ms", r.timings.ttfb.map(ms));
    a.f64("gateway.ttft_ms", r.timings.ttft.map(ms));
    a.f64("gateway.ttft_text_ms", r.timings.ttft_text.map(ms));
    a.f64("gateway.output_tps", r.timings.output_tps);
    a.int("gateway.reasoning_tokens", r.usage.reasoning_tokens);
    a.f64("gateway.cost_usd", r.cost_usd);
    a.str("gateway.caller_id", r.principal.caller_id.as_deref());
    a.str("session.id", r.principal.session_id.as_deref());
    for (k, v) in &r.principal.tags {
        a.str(&format!("gateway.tag.{k}"), Some(v));
    }

    let events: Vec<Value> = [
        ("first_byte", r.timings.ttfb),
        ("first_token", r.timings.ttft),
    ]
    .into_iter()
    .filter_map(|(name, at)| Some(json!({ "name": name, "timeUnixNano": nanos(start, at?) })))
    .collect();

    let mut span = json!({
        "traceId": hex::encode(r.otel_trace_id()),
        "spanId": hex::encode(r.otel_span_id()),
        "name": span_name(r),
        "kind": SPAN_KIND_CLIENT,
        "startTimeUnixNano": nanos(start, Duration::ZERO),
        "endTimeUnixNano": nanos(start, r.timings.duration),
        "attributes": a.0,
        "events": events,
    });
    if let Some(p) = r.trace_parent {
        span["parentSpanId"] = hex::encode(p.parent_span_id).into();
    }
    if matches!(r.outcome, Outcome::UpstreamError | Outcome::GatewayError) {
        span["status"] = json!({
            "code": STATUS_ERROR,
            "message": r.error_type.as_deref().unwrap_or(r.outcome.as_str()),
        });
    }
    span
}

/// `chat {model}` per the `gen_ai` conventions; unobserved paths by method and
/// upstream only, to keep names low-cardinality.
fn span_name(r: &RequestRecord) -> String {
    match (
        r.operation,
        r.requested_model.as_deref().or(r.response_model.as_deref()),
    ) {
        (Some(_), Some(model)) => format!("chat {model}"),
        (Some(_), None) => "chat".into(),
        (None, _) => format!("{} /{}", r.method, r.upstream),
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// OTLP JSON encodes 64-bit integers as strings.
fn nanos(start: SystemTime, offset: Duration) -> String {
    (start + offset)
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .to_string()
}

fn attr(key: &str, value: Value) -> Value {
    Value::Object(Map::from_iter([
        ("key".to_owned(), key.into()),
        ("value".to_owned(), value),
    ]))
}

/// Absent values are skipped.
#[derive(Default)]
struct Attrs(Vec<Value>);

impl Attrs {
    fn push(&mut self, key: &str, value: Value) {
        self.0.push(attr(key, value));
    }

    fn str(&mut self, key: &str, v: Option<&str>) {
        if let Some(v) = v {
            self.push(key, json!({ "stringValue": v }));
        }
    }

    fn int(&mut self, key: &str, v: Option<u64>) {
        if let Some(v) = v {
            self.push(key, json!({ "intValue": v.to_string() }));
        }
    }

    fn f64(&mut self, key: &str, v: Option<f64>) {
        if let Some(v) = v.filter(|v| v.is_finite()) {
            self.push(key, json!({ "doubleValue": v }));
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;

    use uuid::Uuid;

    use super::*;
    use crate::identity::Principal;
    use crate::observe::timings::Timings;
    use crate::providers::{Operation, ProviderKind, RateLimitSnapshot, Usage};
    use crate::trace_context::TraceParent;

    fn record() -> RequestRecord {
        RequestRecord {
            id: Uuid::parse_str("01990000-0000-7000-8000-0000000000aa").unwrap(),
            started_at: UNIX_EPOCH + Duration::from_secs(1_000),
            upstream: "anthropic".into(),
            provider: ProviderKind::Anthropic,
            method: "POST".into(),
            path: "/v1/messages".into(),
            operation: Some(Operation::Messages),
            requested_model: Some("claude-sonnet-5".into()),
            response_model: Some("claude-sonnet-5-20260901".into()),
            streaming: true,
            principal: Principal {
                caller_id: Some("c1".into()),
                session_id: Some("s1".into()),
                tags: BTreeMap::from([("project".into(), "ig".into())]),
            },
            trace_parent: None,
            http_status: Some(200),
            outcome: Outcome::Ok,
            error_type: None,
            stop_reason: Some("end_turn".into()),
            usage: Usage {
                input_tokens_total: Some(100),
                output_tokens: Some(20),
                cache_read_tokens: Some(80),
                ..Usage::default()
            },
            timings: Timings {
                ttfb: Some(Duration::from_millis(300)),
                ttft: Some(Duration::from_millis(500)),
                duration: Duration::from_secs(2),
                ..Timings::default()
            },
            rate_limits: RateLimitSnapshot::default(),
            cost_usd: Some(0.01),
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

    #[test]
    fn span_carries_ids_times_and_genai_attributes() {
        let s = span(&record());
        assert_eq!(s["traceId"], "019900000000700080000000000000aa");
        assert_eq!(s["spanId"], "80000000000000aa");
        assert_eq!(s["name"], "chat claude-sonnet-5");
        assert_eq!(s["kind"], 3);
        assert_eq!(s["startTimeUnixNano"], "1000000000000");
        assert_eq!(s["endTimeUnixNano"], "1002000000000");
        assert!(s.get("parentSpanId").is_none());
        assert!(s.get("status").is_none());
        assert_eq!(
            s["events"],
            json!([
                { "name": "first_byte", "timeUnixNano": "1000300000000" },
                { "name": "first_token", "timeUnixNano": "1000500000000" },
            ])
        );
        let a = attrs(&s);
        assert_eq!(a["gen_ai.operation.name"]["stringValue"], "chat");
        assert_eq!(a["gen_ai.provider.name"]["stringValue"], "anthropic");
        assert_eq!(a["gen_ai.request.model"]["stringValue"], "claude-sonnet-5");
        assert_eq!(
            a["gen_ai.response.model"]["stringValue"],
            "claude-sonnet-5-20260901"
        );
        assert_eq!(a["gen_ai.usage.input_tokens"]["intValue"], "100");
        assert_eq!(a["gen_ai.usage.output_tokens"]["intValue"], "20");
        assert_eq!(a["gen_ai.usage.cache_read.input_tokens"]["intValue"], "80");
        assert_eq!(
            a["gen_ai.response.finish_reasons"]["arrayValue"]["values"][0]["stringValue"],
            "end_turn"
        );
        assert_eq!(a["gateway.ttft_ms"]["doubleValue"], 500.0);
        assert_eq!(a["gateway.cost_usd"]["doubleValue"], 0.01);
        assert_eq!(a["session.id"]["stringValue"], "s1");
        assert_eq!(a["gateway.tag.project"]["stringValue"], "ig");
        assert!(!a.contains_key("gen_ai.usage.cache_creation.input_tokens"));
        assert!(!a.contains_key("error.type"));
    }

    #[test]
    fn traceparent_sets_trace_and_parent() {
        let mut r = record();
        r.trace_parent = Some(TraceParent {
            trace_id: [0xab; 16],
            parent_span_id: [0xcd; 8],
            sampled: true,
        });
        let s = span(&r);
        assert_eq!(s["traceId"], "ab".repeat(16));
        assert_eq!(s["parentSpanId"], "cd".repeat(8));
        assert_eq!(s["spanId"], "80000000000000aa");
    }

    #[test]
    fn errors_set_status() {
        let mut r = record();
        r.outcome = Outcome::UpstreamError;
        r.error_type = Some("overloaded_error".into());
        let s = span(&r);
        assert_eq!(s["status"]["code"], 2);
        assert_eq!(s["status"]["message"], "overloaded_error");
        assert_eq!(attrs(&s)["error.type"]["stringValue"], "overloaded_error");

        r.outcome = Outcome::ClientCancelled;
        r.error_type = None;
        assert!(span(&r).get("status").is_none());
    }

    #[test]
    fn unobserved_path_is_named_by_method_and_upstream() {
        let mut r = record();
        r.operation = None;
        r.requested_model = None;
        r.response_model = None;
        let s = span(&r);
        assert_eq!(s["name"], "POST /anthropic");
        assert!(!attrs(&s).contains_key("gen_ai.operation.name"));
    }
}
