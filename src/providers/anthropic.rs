//! Anthropic Messages API. `input_tokens` excludes cache tokens; `output_tokens`
//! in `message_delta` is cumulative. Claude Code hides thinking text: its
//! `thinking_delta`s are empty, so they count as thinking regardless.

use axum::http::{HeaderMap, Method};
use serde::Deserialize;

use super::{
    DeltaKind, Operation, ProviderAdapter, ProviderKind, RateLimitSnapshot, RequestInfo,
    StreamEvent, StreamParser, Usage, header_f64, header_u64, inspect_common, retry_after,
};
use crate::observe::sse::{SseEvent, SseParser};

#[derive(Debug)]
pub struct Anthropic;

impl ProviderAdapter for Anthropic {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Anthropic
    }

    fn classify(&self, method: &Method, path: &str) -> Option<Operation> {
        (method == Method::POST && path.ends_with("/v1/messages")).then_some(Operation::Messages)
    }

    fn inspect_request(&self, _op: Operation, body: &[u8]) -> RequestInfo {
        inspect_common(body)
    }

    fn stream_parser(&self) -> Box<dyn StreamParser> {
        Box::new(Parser::default())
    }

    fn parse_response(&self, body: &[u8]) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        match serde_json::from_slice::<Response>(body) {
            Ok(Response::Message(m)) => {
                push_message(&mut out, m.model, m.usage);
                if let Some(stop) = m.stop_reason {
                    out.push(StreamEvent::Stop(stop));
                }
            }
            Ok(Response::Error { error }) => out.push(error.into()),
            Err(_) => {}
        }
        out
    }

    fn rate_limits(&self, headers: &HeaderMap) -> RateLimitSnapshot {
        RateLimitSnapshot {
            requests_remaining: header_u64(headers, "anthropic-ratelimit-requests-remaining"),
            tokens_remaining: header_u64(headers, "anthropic-ratelimit-tokens-remaining"),
            input_tokens_remaining: header_u64(
                headers,
                "anthropic-ratelimit-input-tokens-remaining",
            ),
            output_tokens_remaining: header_u64(
                headers,
                "anthropic-ratelimit-output-tokens-remaining",
            ),
            retry_after: retry_after(headers),
            window_5h_utilization: header_f64(
                headers,
                "anthropic-ratelimit-unified-5h-utilization",
            ),
            window_7d_utilization: header_f64(
                headers,
                "anthropic-ratelimit-unified-7d-utilization",
            ),
        }
    }
}

#[derive(Default)]
struct Parser {
    sse: SseParser,
    events: Vec<SseEvent>,
}

impl StreamParser for Parser {
    fn push(&mut self, chunk: &[u8], out: &mut Vec<StreamEvent>) {
        self.sse.push(chunk, &mut self.events);
        self.drain(out);
    }

    fn finish(&mut self, out: &mut Vec<StreamEvent>) {
        self.sse.finish(&mut self.events);
        self.drain(out);
    }
}

impl Parser {
    fn drain(&mut self, out: &mut Vec<StreamEvent>) {
        for ev in self.events.drain(..) {
            let Ok(ev) = serde_json::from_str::<Event>(&ev.data) else {
                continue;
            };
            match ev {
                Event::MessageStart { message } => push_message(out, message.model, message.usage),
                Event::ContentBlockDelta { delta } => {
                    if let Some(kind) = delta.kind() {
                        out.push(StreamEvent::ContentDelta(kind));
                    }
                }
                Event::MessageDelta { delta, usage } => {
                    if let Some(stop) = delta.stop_reason {
                        out.push(StreamEvent::Stop(stop));
                    }
                    if let Some(u) = usage {
                        out.push(StreamEvent::Usage(u.normalize()));
                    }
                }
                Event::Error { error } => out.push(error.into()),
                Event::Other => {}
            }
        }
    }
}

fn push_message(out: &mut Vec<StreamEvent>, model: Option<String>, usage: Option<RawUsage>) {
    if let Some(m) = model {
        out.push(StreamEvent::ResponseModel(m));
    }
    if let Some(u) = usage {
        out.push(StreamEvent::Usage(u.normalize()));
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Event {
    MessageStart {
        message: Message,
    },
    ContentBlockDelta {
        delta: Delta,
    },
    MessageDelta {
        delta: MessageDeltaBody,
        usage: Option<RawUsage>,
    },
    Error {
        error: ErrorBody,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Response {
    Message(Message),
    Error { error: ErrorBody },
}

#[derive(Deserialize)]
struct Message {
    model: Option<String>,
    usage: Option<RawUsage>,
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum Delta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "thinking_delta", alias = "signature_delta")]
    Thinking,
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
    #[serde(other)]
    Other,
}

impl Delta {
    fn kind(&self) -> Option<DeltaKind> {
        match self {
            Self::Text { text } if !text.is_empty() => Some(DeltaKind::Text),
            Self::Thinking => Some(DeltaKind::Thinking),
            Self::InputJson { partial_json } if !partial_json.is_empty() => {
                Some(DeltaKind::ToolInput)
            }
            _ => None,
        }
    }
}

#[derive(Deserialize)]
struct MessageDeltaBody {
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct ErrorBody {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    message: String,
}

impl From<ErrorBody> for StreamEvent {
    fn from(e: ErrorBody) -> Self {
        Self::Error {
            kind: e.kind,
            message: e.message,
        }
    }
}

#[derive(Deserialize)]
#[allow(clippy::struct_field_names)]
struct RawUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    cache_creation: Option<CacheCreation>,
    output_tokens_details: Option<OutputDetails>,
}

#[derive(Deserialize)]
struct CacheCreation {
    ephemeral_1h_input_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct OutputDetails {
    thinking_tokens: Option<u64>,
}

impl RawUsage {
    /// Cache fields default to 0 only when `input_tokens` is present, so a
    /// `message_delta` carrying just `output_tokens` leaves the input side alone.
    fn normalize(self) -> Usage {
        let (total, read, write) = match self.input_tokens {
            Some(input) => {
                let read = self.cache_read_input_tokens.unwrap_or(0);
                let write = self.cache_creation_input_tokens.unwrap_or(0);
                (Some(input + read + write), Some(read), Some(write))
            }
            None => (
                None,
                self.cache_read_input_tokens,
                self.cache_creation_input_tokens,
            ),
        };
        Usage {
            input_tokens_total: total,
            output_tokens: self.output_tokens,
            cache_read_tokens: read,
            cache_write_tokens: write,
            // `message_delta` repeats the cache totals without this breakdown.
            cache_write_1h_tokens: self
                .cache_creation
                .and_then(|c| c.ephemeral_1h_input_tokens),
            reasoning_tokens: self.output_tokens_details.and_then(|d| d.thinking_tokens),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::providers::testing::{
        deltas, headers, parse_all, split_at, sse_fixtures, total_usage,
    };

    const TEXT_SSE: &[u8] = include_bytes!("../../tests/fixtures/anthropic/text.sse");
    const TEXT_JSON: &[u8] = include_bytes!("../../tests/fixtures/anthropic/text.json");
    const ERROR_SSE: &[u8] = include_bytes!("../../tests/fixtures/anthropic/error.sse");
    const RATE_LIMITED_JSON: &[u8] =
        include_bytes!("../../tests/fixtures/anthropic/rate_limited.json");
    const RATE_LIMITED_HEADERS: &str =
        include_str!("../../tests/fixtures/anthropic/rate_limited.headers");

    #[test]
    fn classifies_messages_only() {
        assert_eq!(
            Anthropic.classify(&Method::POST, "/v1/messages"),
            Some(Operation::Messages)
        );
        assert_eq!(Anthropic.classify(&Method::GET, "/v1/messages"), None);
        assert_eq!(
            Anthropic.classify(&Method::POST, "/v1/messages/count_tokens"),
            None
        );
    }

    #[test]
    fn inspects_request() {
        let info = Anthropic.inspect_request(
            Operation::Messages,
            br#"{"model":"claude-sonnet-5","max_tokens":64,"stream":true,"messages":[]}"#,
        );
        assert_eq!(info.model.as_deref(), Some("claude-sonnet-5"));
        assert!(info.stream);
        assert_eq!(info.max_tokens, Some(64));
        assert_eq!(
            Anthropic.inspect_request(Operation::Messages, b"not json"),
            RequestInfo::default()
        );
    }

    #[test]
    fn parses_text_stream() {
        let events = parse_all(&Anthropic, &[TEXT_SSE]);
        assert!(matches!(&events[0], StreamEvent::ResponseModel(_)));
        assert!(events.contains(&StreamEvent::ContentDelta(DeltaKind::Text)));
        assert!(events.contains(&StreamEvent::Stop("end_turn".into())));
        let u = total_usage(&events);
        assert!(u.input_tokens_total.is_some());
        assert!(u.output_tokens.is_some());
    }

    #[test]
    fn parses_mid_stream_error() {
        let events = parse_all(&Anthropic, &[ERROR_SSE]);
        assert_eq!(
            events.last(),
            Some(&StreamEvent::Error {
                kind: "overloaded_error".into(),
                message: "Overloaded".into()
            })
        );
    }

    #[test]
    fn non_stream_matches_stream_shape() {
        let events = Anthropic.parse_response(TEXT_JSON);
        assert!(matches!(&events[0], StreamEvent::ResponseModel(_)));
        assert!(events.contains(&StreamEvent::Stop("end_turn".into())));
        assert!(total_usage(&events).output_tokens.is_some());
    }

    #[test]
    fn parses_error_body() {
        let events = Anthropic.parse_response(RATE_LIMITED_JSON);
        assert!(
            matches!(&events[..], [StreamEvent::Error { kind, .. }] if kind == "rate_limit_error")
        );
    }

    #[test]
    fn normalizes_cache_tokens() {
        let raw: RawUsage = serde_json::from_str(
            r#"{"input_tokens":10,"cache_read_input_tokens":1000,"cache_creation_input_tokens":200,"output_tokens":5}"#,
        )
        .unwrap();
        let u = raw.normalize();
        assert_eq!(u.input_tokens_total, Some(1210));
        assert_eq!(u.cache_read_tokens, Some(1000));
        assert_eq!(u.cache_write_tokens, Some(200));
        assert_eq!(u.cache_write_1h_tokens, None);

        let mut total = u;
        total.merge(
            &serde_json::from_str::<RawUsage>(r#"{"output_tokens":42}"#)
                .unwrap()
                .normalize(),
        );
        assert_eq!(total.input_tokens_total, Some(1210));
        assert_eq!(total.output_tokens, Some(42));
    }

    #[test]
    fn reads_rate_limit_headers() {
        let rl = Anthropic.rate_limits(&headers(RATE_LIMITED_HEADERS));
        assert_eq!(rl.requests_remaining, Some(0));
        assert_eq!(rl.tokens_remaining, Some(81000));
        assert_eq!(rl.input_tokens_remaining, Some(45000));
        assert_eq!(rl.output_tokens_remaining, Some(36000));
        assert_eq!(rl.retry_after, Some(std::time::Duration::from_secs(12)));
    }

    #[test]
    fn thinking_comes_before_text() {
        let events = parse_all(&Anthropic, &[&fixture("thinking.sse")]);
        assert_eq!(deltas(&events), [DeltaKind::Thinking, DeltaKind::Text]);
    }

    #[test]
    fn claude_code_hidden_thinking_still_counts() {
        let events = parse_all(&Anthropic, &[&fixture("cc_thinking.sse")]);
        assert_eq!(deltas(&events), [DeltaKind::Thinking, DeltaKind::Text]);
        assert_eq!(total_usage(&events).reasoning_tokens, Some(32));
    }

    #[test]
    fn tool_use_is_tool_input() {
        let events = parse_all(&Anthropic, &[&fixture("cc_tool_use.sse")]);
        assert_eq!(deltas(&events), [DeltaKind::Thinking, DeltaKind::ToolInput]);
        assert!(events.contains(&StreamEvent::Stop("tool_use".into())));
        assert_eq!(total_usage(&events).reasoning_tokens, Some(28));
    }

    #[test]
    fn cache_hit_counts_cache_in_total() {
        let u = total_usage(&parse_all(&Anthropic, &[&fixture("cc_cache_hit.sse")]));
        assert_eq!(u.input_tokens_total, Some(2 + 221 + 92_467));
        assert_eq!(u.cache_read_tokens, Some(92_467));
        assert_eq!(u.cache_write_tokens, Some(221));
        assert_eq!(u.cache_write_1h_tokens, Some(221));
        assert_eq!(u.output_tokens, Some(6));
    }

    #[test]
    fn cache_write_counts_cache_in_total() {
        let u = total_usage(&parse_all(&Anthropic, &[&fixture("cc_cache_write.sse")]));
        assert_eq!(u.input_tokens_total, Some(2 + 91_176));
        assert_eq!(u.cache_read_tokens, Some(0));
        assert_eq!(u.cache_write_tokens, Some(91_176));
        assert_eq!(u.cache_write_1h_tokens, Some(91_176));
        assert_eq!(u.output_tokens, Some(6));
    }

    #[test]
    fn claude_code_non_stream_and_usage_limit() {
        let events = Anthropic.parse_response(&fixture("cc_text.json"));
        assert!(events.contains(&StreamEvent::Stop("max_tokens".into())));
        assert_eq!(total_usage(&events).output_tokens, Some(1));
        let events = Anthropic.parse_response(&fixture("cc_usage_limit.json"));
        assert!(
            matches!(&events[..], [StreamEvent::Error { kind, .. }] if kind == "rate_limit_error")
        );
    }

    #[test]
    fn reads_subscription_windows() {
        let raw = String::from_utf8(fixture("cc_text.headers")).unwrap();
        let rl = Anthropic.rate_limits(&headers(&raw));
        assert_eq!(rl.window_5h_utilization, Some(0.13));
        assert_eq!(rl.window_7d_utilization, Some(0.07));
        assert_eq!(rl.requests_remaining, None);
    }

    #[test]
    fn subscription_usage_limit_has_no_rate_limit_headers() {
        let raw = String::from_utf8(fixture("cc_usage_limit.headers")).unwrap();
        assert_eq!(
            Anthropic.rate_limits(&headers(&raw)),
            RateLimitSnapshot::default()
        );
    }

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(format!(
            "{}/tests/fixtures/anthropic/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    }

    proptest! {
        #[test]
        fn random_splits_match(cuts in proptest::collection::vec(any::<usize>(), 0..30)) {
            for (name, fixture) in sse_fixtures("anthropic") {
                let whole = parse_all(&Anthropic, &[&fixture]);
                let chunks = split_at(&fixture, &cuts);
                let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
                prop_assert_eq!(parse_all(&Anthropic, &refs), whole, "{}", name);
            }
        }
    }
}
