//! One record per request, including failed and cancelled ones.

use std::time::SystemTime;

use bytes::Bytes;
use uuid::Uuid;

use super::timings::Timings;
use crate::identity::Principal;
use crate::providers::{Operation, ProviderKind, RateLimitSnapshot, Usage};
use crate::trace_context::TraceParent;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Ok,
    UpstreamError,
    ClientCancelled,
    GatewayError,
}

impl Outcome {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::UpstreamError => "upstream_error",
            Self::ClientCancelled => "client_cancelled",
            Self::GatewayError => "gateway_error",
        }
    }
}

#[derive(Debug, Clone)]
pub struct RequestRecord {
    /// `UUIDv7`; also the trace id.
    pub id: Uuid,
    pub started_at: SystemTime,
    pub upstream: String,
    pub provider: ProviderKind,
    pub method: String,
    /// Upstream-relative, without the query.
    pub path: String,
    /// `None` for unobserved paths.
    pub operation: Option<Operation>,
    pub requested_model: Option<String>,
    pub response_model: Option<String>,
    pub streaming: bool,
    pub principal: Principal,
    /// Client's W3C `traceparent`, if valid.
    pub trace_parent: Option<TraceParent>,
    /// Status sent to the client; `None` if cancelled before upstream answered.
    pub http_status: Option<u16>,
    pub outcome: Outcome,
    pub error_type: Option<String>,
    pub stop_reason: Option<String>,
    pub usage: Usage,
    pub timings: Timings,
    pub rate_limits: RateLimitSnapshot,
    /// Set by the `cost` module; `None` for unpriced models.
    pub cost_usd: Option<f64>,
}

impl RequestRecord {
    /// The model that answered, else the one requested.
    #[must_use]
    pub fn model(&self) -> Option<&str> {
        self.response_model
            .as_deref()
            .or(self.requested_model.as_deref())
    }

    /// The caller's trace when it sent a `traceparent`, else the record id.
    #[must_use]
    pub fn otel_trace_id(&self) -> [u8; 16] {
        self.trace_parent
            .map_or(*self.id.as_bytes(), |p| p.trace_id)
    }

    /// The random tail of the `UUIDv7`, so the span id is stable per record.
    #[must_use]
    pub fn otel_span_id(&self) -> [u8; 8] {
        let mut span = [0; 8];
        span.copy_from_slice(&self.id.as_bytes()[8..]);
        span
    }
}

/// What export modules receive: the record plus captured content, if any.
#[derive(Debug, Clone)]
pub struct CompletedExchange {
    pub record: RequestRecord,
    pub content: Option<CapturedContent>,
}

#[derive(Debug, Clone)]
pub struct CapturedContent {
    /// As sent by the client, before any gateway rewrite.
    pub request_body: Bytes,
    /// Raw upstream bytes (SSE or JSON), at most `capture.max_bytes`.
    pub raw_response: Bytes,
    pub truncated: bool,
}
