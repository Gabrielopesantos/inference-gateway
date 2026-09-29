//! Observation of proxied exchanges. Bytes reach the client unchanged; a copy
//! feeds the adapter's parser. Every [`Observation`] emits exactly one
//! [`RequestRecord`]: on completion, failure, or `Drop` (client cancel).

pub mod record;
pub mod sse;
pub mod timings;

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::SystemTime;

use axum::http::{HeaderMap, Method, StatusCode, header};
use bytes::{Bytes, BytesMut};
use futures_util::Stream;
use tokio::time::Instant;
use uuid::Uuid;

use self::record::{Outcome, RequestRecord};
use self::timings::Clock;
use crate::error::GatewayError;
use crate::identity::Principal;
use crate::providers::{
    Operation, ProviderAdapter, RateLimitSnapshot, RequestInfo, StreamEvent, StreamParser, Usage,
};

/// Receives each finished record. Called on the proxy path, so it must not block.
pub type RecordSink = Arc<dyn Fn(RequestRecord) + Send + Sync>;

/// Non-stream bodies above this are passed through unparsed.
const MAX_JSON_TEE: usize = 8 * 1024 * 1024;

enum Parser {
    None,
    Stream(Box<dyn StreamParser>),
    Json { buf: BytesMut, overflow: bool },
}

#[derive(Clone, Copy)]
enum End {
    Complete,
    StreamError,
    Cancelled,
    Gateway(&'static str),
}

pub struct Observation {
    id: Uuid,
    started_at: SystemTime,
    upstream: String,
    adapter: Arc<dyn ProviderAdapter>,
    sink: RecordSink,
    finished: bool,

    principal: Principal,
    method: Method,
    path: String,
    operation: Option<Operation>,
    request: RequestInfo,

    clock: Clock,
    status: Option<StatusCode>,
    streaming: Option<bool>,
    rate_limits: RateLimitSnapshot,
    parser: Parser,
    expected_len: Option<u64>,
    received: u64,

    events: Vec<StreamEvent>,
    usage: Usage,
    response_model: Option<String>,
    stop_reason: Option<String>,
    error_type: Option<String>,
}

impl Observation {
    #[must_use]
    pub fn new(
        upstream: &str,
        adapter: Arc<dyn ProviderAdapter>,
        sink: RecordSink,
        start: Instant,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            started_at: SystemTime::now(),
            upstream: upstream.to_owned(),
            adapter,
            sink,
            finished: false,
            principal: Principal::default(),
            method: Method::GET,
            path: String::new(),
            operation: None,
            request: RequestInfo::default(),
            clock: Clock::new(start),
            status: None,
            streaming: None,
            rate_limits: RateLimitSnapshot::default(),
            parser: Parser::None,
            expected_len: None,
            received: 0,
            events: Vec::new(),
            usage: Usage::default(),
            response_model: None,
            stop_reason: None,
            error_type: None,
        }
    }

    pub fn target(&mut self, method: &Method, path: &str) {
        self.method = method.clone();
        path.clone_into(&mut self.path);
    }

    pub fn principal(&mut self, principal: Principal) {
        self.principal = principal;
    }

    pub fn request(&mut self, operation: Operation, info: RequestInfo) {
        self.operation = Some(operation);
        self.request = info;
    }

    /// Upstream status and headers arrived.
    pub fn response(&mut self, status: StatusCode, headers: &HeaderMap) {
        self.clock.headers(Instant::now());
        self.status = Some(status);
        self.rate_limits = self.adapter.rate_limits(headers);
        self.expected_len = headers
            .get(header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok()?.parse().ok());
        let sse = headers
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.starts_with("text/event-stream"));
        self.streaming = Some(sse);
        if self.operation.is_some() {
            self.parser = if sse {
                Parser::Stream(self.adapter.stream_parser())
            } else {
                Parser::Json {
                    buf: BytesMut::new(),
                    overflow: false,
                }
            };
        }
        // No body follows, and hyper may drop it without a poll.
        if self.method == Method::HEAD
            || status == StatusCode::NO_CONTENT
            || status == StatusCode::NOT_MODIFIED
        {
            self.finish(End::Complete);
        }
    }

    pub fn into_body<S, E>(self, inner: S) -> ObservedBody<S>
    where
        S: Stream<Item = Result<Bytes, E>>,
    {
        ObservedBody {
            inner: Box::pin(inner),
            obs: self,
        }
    }

    /// The gateway answered with `err` instead of forwarding.
    pub fn fail(mut self, err: &GatewayError) {
        self.status = Some(err.status());
        self.finish(End::Gateway(err.kind()));
    }

    fn chunk(&mut self, bytes: &[u8]) {
        let now = Instant::now();
        self.clock.chunk(now);
        self.received += bytes.len() as u64;
        match &mut self.parser {
            Parser::None => {}
            Parser::Stream(p) => {
                p.push(bytes, &mut self.events);
                self.apply_events(now);
            }
            Parser::Json { buf, overflow } => {
                if buf.len() + bytes.len() > MAX_JSON_TEE {
                    *overflow = true;
                    *buf = BytesMut::new();
                } else if !*overflow {
                    buf.extend_from_slice(bytes);
                }
            }
        }
        // hyper may drop a fixed-length body without polling its end.
        if self.expected_len.is_some_and(|len| self.received >= len) {
            self.finish(End::Complete);
        }
    }

    fn apply_events(&mut self, now: Instant) {
        let mut events = std::mem::take(&mut self.events);
        for ev in events.drain(..) {
            match ev {
                StreamEvent::ResponseModel(m) => {
                    self.response_model.get_or_insert(m);
                }
                StreamEvent::ContentDelta(kind) => self.clock.delta(kind, now),
                StreamEvent::Usage(u) => self.usage.merge(&u),
                StreamEvent::Stop(s) => self.stop_reason = Some(s),
                StreamEvent::Error { kind, .. } => {
                    self.error_type.get_or_insert(kind);
                }
            }
        }
        self.events = events;
    }

    fn finish(&mut self, end: End) {
        if self.finished {
            return;
        }
        self.finished = true;
        let now = Instant::now();

        match std::mem::replace(&mut self.parser, Parser::None) {
            Parser::None => {}
            Parser::Stream(mut p) => {
                p.finish(&mut self.events);
                self.apply_events(now);
            }
            Parser::Json { buf, overflow } => {
                if !overflow && !buf.is_empty() {
                    self.events = self.adapter.parse_response(&buf);
                    self.apply_events(now);
                }
            }
        }

        let failed_status = self
            .status
            .filter(|s| s.is_client_error() || s.is_server_error());
        let (outcome, error_type) = match end {
            End::Gateway(kind) => (Outcome::GatewayError, Some(kind.to_owned())),
            End::Cancelled => (Outcome::ClientCancelled, self.error_type.take()),
            End::StreamError => (
                Outcome::UpstreamError,
                Some(
                    self.error_type
                        .take()
                        .unwrap_or_else(|| "upstream_stream_error".into()),
                ),
            ),
            End::Complete => match (self.error_type.take(), failed_status) {
                (Some(kind), _) => (Outcome::UpstreamError, Some(kind)),
                (None, Some(s)) => (Outcome::UpstreamError, Some(format!("http_{}", s.as_u16()))),
                (None, None) => (Outcome::Ok, None),
            },
        };

        let record = RequestRecord {
            id: self.id,
            started_at: self.started_at,
            upstream: std::mem::take(&mut self.upstream),
            provider: self.adapter.kind(),
            method: self.method.to_string(),
            path: std::mem::take(&mut self.path),
            operation: self.operation,
            requested_model: self.request.model.take(),
            response_model: self.response_model.take(),
            streaming: self.streaming.unwrap_or(self.request.stream),
            principal: std::mem::take(&mut self.principal),
            http_status: self.status.map(|s| s.as_u16()),
            outcome,
            error_type,
            stop_reason: self.stop_reason.take(),
            usage: self.usage,
            timings: self.clock.finish(now, self.usage.output_tokens),
            rate_limits: self.rate_limits,
            cost_usd: None,
        };
        (self.sink)(record);
    }
}

impl Drop for Observation {
    fn drop(&mut self) {
        self.finish(End::Cancelled);
    }
}

/// Response body wrapper: yields upstream bytes unchanged and observes a copy.
pub struct ObservedBody<S> {
    inner: Pin<Box<S>>,
    obs: Observation,
}

impl<S, E> Stream for ObservedBody<S>
where
    S: Stream<Item = Result<Bytes, E>>,
{
    type Item = Result<Bytes, E>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let item = this.inner.as_mut().poll_next(cx);
        match &item {
            Poll::Ready(Some(Ok(bytes))) => this.obs.chunk(bytes),
            Poll::Ready(Some(Err(_))) => this.obs.finish(End::StreamError),
            Poll::Ready(None) => this.obs.finish(End::Complete),
            Poll::Pending => {}
        }
        item
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::convert::Infallible;
    use std::sync::Mutex;
    use std::time::Duration;

    use futures_util::StreamExt;
    use futures_util::stream;

    use super::*;
    use crate::providers::anthropic::Anthropic;

    const TEXT_SSE: &[u8] = include_bytes!("../../tests/fixtures/anthropic/text.sse");
    const THINKING_SSE: &[u8] = include_bytes!("../../tests/fixtures/anthropic/thinking.sse");
    const TEXT_JSON: &[u8] = include_bytes!("../../tests/fixtures/anthropic/text.json");

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    fn collector() -> (RecordSink, Arc<Mutex<Vec<RequestRecord>>>) {
        let records = Arc::new(Mutex::new(Vec::new()));
        let sink_records = records.clone();
        let sink: RecordSink = Arc::new(move |r| sink_records.lock().unwrap().push(r));
        (sink, records)
    }

    fn observation(sink: RecordSink) -> Observation {
        let mut obs = Observation::new("anthropic", Arc::new(Anthropic), sink, Instant::now());
        obs.request(
            Operation::Messages,
            RequestInfo {
                model: Some("claude-sonnet-5".into()),
                stream: true,
                max_tokens: Some(64),
            },
        );
        obs
    }

    fn sse_headers() -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(header::CONTENT_TYPE, "text/event-stream".parse().unwrap());
        h
    }

    fn events(fixture: &[u8]) -> Vec<Bytes> {
        fixture
            .split_inclusive(|&b| b == b'\n')
            .collect::<Vec<_>>()
            .split(|line| *line == b"\n")
            .filter(|lines| !lines.is_empty())
            .map(|lines| {
                let mut ev = lines.concat();
                ev.push(b'\n');
                Bytes::from(ev)
            })
            .collect()
    }

    /// `header_delay`, then one event per `gap`.
    fn scripted(
        chunks: Vec<Bytes>,
        gap: Duration,
    ) -> impl Stream<Item = Result<Bytes, Infallible>> {
        stream::iter(chunks).then(move |c| async move {
            tokio::time::sleep(gap).await;
            Ok(c)
        })
    }

    async fn run(fixture: &[u8]) -> RequestRecord {
        let (sink, records) = collector();
        let mut obs = observation(sink);
        tokio::time::sleep(ms(100)).await;
        obs.response(StatusCode::OK, &sse_headers());
        let mut body = obs.into_body(scripted(events(fixture), ms(10)));
        while body.next().await.is_some() {}
        drop(body);
        let mut records = records.lock().unwrap();
        assert_eq!(records.len(), 1);
        records.pop().unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn exact_timings_for_text_stream() {
        // 11 events: start, block start, ping, 5 text deltas, block stop, delta, stop.
        let r = run(TEXT_SSE).await;
        assert_eq!(r.outcome, Outcome::Ok);
        assert_eq!(r.timings.ttfb, Some(ms(100)));
        assert_eq!(r.timings.ttft, Some(ms(140)));
        assert_eq!(r.timings.ttft_text, Some(ms(140)));
        assert_eq!(r.timings.duration, ms(210));
        assert_eq!(r.timings.chunk_count, 11);
        assert_eq!(r.timings.chunk_gap_mean, Some(ms(10)));
        assert_eq!(r.timings.chunk_gap_max, Some(ms(10)));
        let tps = r.timings.output_tps.unwrap();
        assert!((tps - 12.0 / 0.070).abs() < 1e-6, "{tps}");
        assert_eq!(r.usage.output_tokens, Some(12));
        assert_eq!(r.response_model.as_deref(), Some("claude-sonnet-5"));
        assert_eq!(r.stop_reason.as_deref(), Some("end_turn"));
        assert!(r.streaming);
    }

    #[tokio::test(start_paused = true)]
    async fn thinking_precedes_text() {
        let r = run(THINKING_SSE).await;
        assert_eq!(r.timings.ttft, Some(ms(130)));
        assert_eq!(r.timings.ttft_text, Some(ms(180)));
        assert!(r.timings.ttft < r.timings.ttft_text);
    }

    #[tokio::test(start_paused = true)]
    async fn drop_mid_stream_is_client_cancelled() {
        let (sink, records) = collector();
        let mut obs = observation(sink);
        obs.response(StatusCode::OK, &sse_headers());
        let mut body = obs.into_body(scripted(events(TEXT_SSE), ms(10)));
        for _ in 0..4 {
            body.next().await.unwrap().unwrap();
        }
        drop(body);
        let records = records.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].outcome, Outcome::ClientCancelled);
        assert_eq!(records[0].timings.duration, ms(40));
        assert!(records[0].usage.input_tokens_total.is_some());
    }

    #[tokio::test(start_paused = true)]
    async fn drop_before_response_is_client_cancelled() {
        let (sink, records) = collector();
        drop(observation(sink));
        let records = records.lock().unwrap();
        assert_eq!(records[0].outcome, Outcome::ClientCancelled);
        assert_eq!(records[0].http_status, None);
    }

    #[tokio::test(start_paused = true)]
    async fn fixed_length_json_completes_without_end_poll() {
        let (sink, records) = collector();
        let mut obs = observation(sink);
        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
        headers.insert(header::CONTENT_LENGTH, TEXT_JSON.len().into());
        obs.response(StatusCode::OK, &headers);
        let (a, b) = TEXT_JSON.split_at(40);
        let mut body = obs.into_body(scripted(
            vec![Bytes::from_static(a), Bytes::from_static(b)],
            ms(5),
        ));
        body.next().await.unwrap().unwrap();
        body.next().await.unwrap().unwrap();
        drop(body);
        let records = records.lock().unwrap();
        assert_eq!(records[0].outcome, Outcome::Ok);
        assert!(!records[0].streaming);
        assert_eq!(records[0].usage.output_tokens, Some(12));
    }

    #[tokio::test(start_paused = true)]
    async fn error_status_without_parsed_error_is_http_status() {
        let (sink, records) = collector();
        let mut obs = observation(sink);
        obs.response(StatusCode::BAD_GATEWAY, &HeaderMap::new());
        let mut body = obs.into_body(scripted(vec![Bytes::from_static(b"<html>")], ms(1)));
        while body.next().await.is_some() {}
        drop(body);
        let records = records.lock().unwrap();
        assert_eq!(records[0].outcome, Outcome::UpstreamError);
        assert_eq!(records[0].error_type.as_deref(), Some("http_502"));
    }

    #[tokio::test(start_paused = true)]
    async fn stream_error_is_upstream_error() {
        let (sink, records) = collector();
        let mut obs = observation(sink);
        obs.response(StatusCode::OK, &sse_headers());
        let items: Vec<Result<Bytes, &str>> =
            vec![Ok(events(TEXT_SSE)[0].clone()), Err("connection reset")];
        let mut body = obs.into_body(stream::iter(items));
        while let Some(item) = body.next().await {
            if item.is_err() {
                break;
            }
        }
        drop(body);
        let records = records.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].outcome, Outcome::UpstreamError);
        assert_eq!(
            records[0].error_type.as_deref(),
            Some("upstream_stream_error")
        );
    }
}
