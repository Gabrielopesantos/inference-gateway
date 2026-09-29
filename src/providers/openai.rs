//! `OpenAI` Chat Completions, also used for compatible servers (vLLM, Ollama,
//! `OpenRouter`). `prompt_tokens` includes cached tokens. Streams report usage
//! only with `stream_options.include_usage`.

use axum::http::{HeaderMap, Method};
use bytes::Bytes;
use serde::Deserialize;
use serde::de::IgnoredAny;
use serde_json::Value;

use super::{
    DeltaKind, Operation, ProviderAdapter, ProviderKind, RateLimitSnapshot, RequestInfo,
    StreamEvent, StreamParser, Usage, header_u64, inspect_common, retry_after,
};
use crate::observe::sse::{SseEvent, SseParser};

#[derive(Debug)]
pub struct OpenAi {
    pub inject_stream_usage: bool,
}

impl ProviderAdapter for OpenAi {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Openai
    }

    fn classify(&self, method: &Method, path: &str) -> Option<Operation> {
        (method == Method::POST && path.ends_with("/chat/completions"))
            .then_some(Operation::ChatCompletions)
    }

    fn inspect_request(&self, _op: Operation, body: &[u8]) -> RequestInfo {
        inspect_common(body)
    }

    /// Adds `stream_options.include_usage = true` to streaming requests that
    /// lack it. Key order is kept; the client then also gets the usage chunk.
    fn rewrite_request(&self, info: &RequestInfo, body: Bytes) -> Bytes {
        if !self.inject_stream_usage || !info.stream {
            return body;
        }
        let Ok(Value::Object(mut req)) = serde_json::from_slice::<Value>(&body) else {
            return body;
        };
        let options = req
            .entry("stream_options")
            .or_insert_with(|| Value::Object(serde_json::Map::new()));
        let Value::Object(options) = options else {
            return body;
        };
        if options.get("include_usage") == Some(&Value::Bool(true)) {
            return body;
        }
        options.insert("include_usage".into(), Value::Bool(true));
        serde_json::to_vec(&req).map_or(body, Bytes::from)
    }

    fn stream_parser(&self) -> Box<dyn StreamParser> {
        Box::new(Parser::default())
    }

    fn parse_response(&self, body: &[u8]) -> Vec<StreamEvent> {
        let mut out = Vec::new();
        if let Ok(resp) = serde_json::from_slice::<Chunk>(body) {
            resp.into_events(true, &mut out);
        }
        out
    }

    fn rate_limits(&self, headers: &HeaderMap) -> RateLimitSnapshot {
        RateLimitSnapshot {
            requests_remaining: header_u64(headers, "x-ratelimit-remaining-requests"),
            tokens_remaining: header_u64(headers, "x-ratelimit-remaining-tokens"),
            input_tokens_remaining: None,
            retry_after: retry_after(headers),
            ..RateLimitSnapshot::default()
        }
    }
}

#[derive(Default)]
struct Parser {
    sse: SseParser,
    events: Vec<SseEvent>,
    model_seen: bool,
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
            if ev.data == "[DONE]" {
                continue;
            }
            if let Ok(chunk) = serde_json::from_str::<Chunk>(&ev.data) {
                let first = !self.model_seen;
                self.model_seen |= chunk.model.is_some();
                chunk.into_events(first, out);
            }
        }
    }
}

/// A stream chunk, a non-stream response, or an error body.
#[derive(Deserialize)]
struct Chunk {
    model: Option<String>,
    #[serde(default)]
    choices: Vec<Choice>,
    usage: Option<RawUsage>,
    error: Option<ErrorBody>,
}

impl Chunk {
    fn into_events(self, with_model: bool, out: &mut Vec<StreamEvent>) {
        if let Some(e) = self.error {
            out.push(StreamEvent::Error {
                kind: e.kind.or(e.code).unwrap_or_else(|| "error".into()),
                message: e.message,
            });
            return;
        }
        if let Some(m) = self.model.filter(|_| with_model) {
            out.push(StreamEvent::ResponseModel(m));
        }
        for choice in self.choices {
            if let Some(kind) = choice.delta.as_ref().and_then(ChoiceDelta::kind) {
                out.push(StreamEvent::ContentDelta(kind));
            }
            if let Some(stop) = choice.finish_reason {
                out.push(StreamEvent::Stop(stop));
            }
        }
        if let Some(u) = self.usage {
            out.push(StreamEvent::Usage(u.normalize()));
        }
    }
}

#[derive(Deserialize)]
struct Choice {
    delta: Option<ChoiceDelta>,
    finish_reason: Option<String>,
}

/// `reasoning_content` (vLLM, `DeepSeek`) and `reasoning` (`OpenRouter`) carry thinking.
#[derive(Deserialize)]
struct ChoiceDelta {
    content: Option<String>,
    refusal: Option<String>,
    reasoning_content: Option<String>,
    reasoning: Option<String>,
    tool_calls: Option<Vec<IgnoredAny>>,
}

impl ChoiceDelta {
    fn kind(&self) -> Option<DeltaKind> {
        let set = |s: &Option<String>| s.as_deref().is_some_and(|s| !s.is_empty());
        if set(&self.content) || set(&self.refusal) {
            Some(DeltaKind::Text)
        } else if set(&self.reasoning_content) || set(&self.reasoning) {
            Some(DeltaKind::Thinking)
        } else if self.tool_calls.as_ref().is_some_and(|t| !t.is_empty()) {
            Some(DeltaKind::ToolInput)
        } else {
            None
        }
    }
}

#[derive(Deserialize)]
struct ErrorBody {
    #[serde(rename = "type")]
    kind: Option<String>,
    code: Option<String>,
    #[serde(default)]
    message: String,
}

#[derive(Deserialize)]
struct RawUsage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    prompt_tokens_details: Option<PromptDetails>,
    completion_tokens_details: Option<CompletionDetails>,
}

#[derive(Deserialize)]
struct PromptDetails {
    cached_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct CompletionDetails {
    reasoning_tokens: Option<u64>,
}

impl RawUsage {
    fn normalize(self) -> Usage {
        let reported = self.prompt_tokens.map(|_| 0);
        Usage {
            input_tokens_total: self.prompt_tokens,
            output_tokens: self.completion_tokens,
            cache_read_tokens: self
                .prompt_tokens_details
                .and_then(|d| d.cached_tokens)
                .or(reported),
            cache_write_tokens: reported,
            cache_write_1h_tokens: reported,
            reasoning_tokens: self
                .completion_tokens_details
                .and_then(|d| d.reasoning_tokens),
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

    const TEXT_SSE: &[u8] = include_bytes!("../../tests/fixtures/openai/text.sse");
    const ERROR_SSE: &[u8] = include_bytes!("../../tests/fixtures/openai/error.sse");
    const RATE_LIMITED_JSON: &[u8] =
        include_bytes!("../../tests/fixtures/openai/rate_limited.json");
    const RATE_LIMITED_HEADERS: &str =
        include_str!("../../tests/fixtures/openai/rate_limited.headers");

    const ADAPTER: OpenAi = OpenAi {
        inject_stream_usage: true,
    };

    fn rewrite(body: &str) -> String {
        let info = ADAPTER.inspect_request(Operation::ChatCompletions, body.as_bytes());
        let out = ADAPTER.rewrite_request(&info, Bytes::copy_from_slice(body.as_bytes()));
        String::from_utf8(out.to_vec()).unwrap()
    }

    #[test]
    fn classifies_chat_completions() {
        assert_eq!(
            ADAPTER.classify(&Method::POST, "/v1/chat/completions"),
            Some(Operation::ChatCompletions)
        );
        assert_eq!(
            ADAPTER.classify(&Method::POST, "/chat/completions"),
            Some(Operation::ChatCompletions)
        );
        assert_eq!(ADAPTER.classify(&Method::GET, "/v1/models"), None);
    }

    #[test]
    fn injects_include_usage_keeping_key_order() {
        assert_eq!(
            rewrite(r#"{"model":"m","stream":true,"messages":[]}"#),
            r#"{"model":"m","stream":true,"messages":[],"stream_options":{"include_usage":true}}"#
        );
        assert_eq!(
            rewrite(r#"{"stream":true,"stream_options":{"include_usage":false}}"#),
            r#"{"stream":true,"stream_options":{"include_usage":true}}"#
        );
    }

    #[test]
    fn leaves_other_requests_untouched() {
        for body in [
            r#"{"model":"m","messages":[]}"#,
            r#"{"stream":true, "stream_options": {"include_usage": true}}"#,
            r#"{"stream":true,"stream_options":null}"#,
            "not json",
        ] {
            assert_eq!(rewrite(body), body);
        }
        let off = OpenAi {
            inject_stream_usage: false,
        };
        let body = Bytes::from_static(br#"{"stream":true}"#);
        let info = off.inspect_request(Operation::ChatCompletions, &body);
        assert_eq!(off.rewrite_request(&info, body.clone()), body);
    }

    #[test]
    fn parses_text_stream() {
        let events = parse_all(&ADAPTER, &[TEXT_SSE]);
        let models = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::ResponseModel(_)))
            .count();
        assert_eq!(models, 1);
        assert!(events.contains(&StreamEvent::ContentDelta(DeltaKind::Text)));
        assert!(events.contains(&StreamEvent::Stop("stop".into())));
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Usage(Usage {
                input_tokens_total: Some(_),
                output_tokens: Some(_),
                ..
            }))
        ));
    }

    #[test]
    fn parses_mid_stream_error() {
        let events = parse_all(&ADAPTER, &[ERROR_SSE]);
        assert!(
            matches!(events.last(), Some(StreamEvent::Error { kind, .. }) if kind == "server_error")
        );
    }

    #[test]
    fn parses_error_body() {
        let events = ADAPTER.parse_response(RATE_LIMITED_JSON);
        assert!(matches!(&events[..], [StreamEvent::Error { kind, .. }] if kind == "requests"));
    }

    #[test]
    fn normalizes_cached_and_reasoning_tokens() {
        let raw: RawUsage = serde_json::from_str(
            r#"{"prompt_tokens":2000,"completion_tokens":300,"prompt_tokens_details":{"cached_tokens":1536},"completion_tokens_details":{"reasoning_tokens":256}}"#,
        )
        .unwrap();
        assert_eq!(
            raw.normalize(),
            Usage {
                input_tokens_total: Some(2000),
                output_tokens: Some(300),
                cache_read_tokens: Some(1536),
                cache_write_tokens: Some(0),
                cache_write_1h_tokens: Some(0),
                reasoning_tokens: Some(256),
            }
        );
    }

    #[test]
    fn reads_rate_limit_headers() {
        let rl = ADAPTER.rate_limits(&headers(RATE_LIMITED_HEADERS));
        assert_eq!(rl.requests_remaining, Some(0));
        assert_eq!(rl.tokens_remaining, Some(199_950));
        assert_eq!(rl.retry_after, Some(std::time::Duration::from_secs(1)));
    }

    #[test]
    fn reasoning_tokens_from_usage() {
        let events = parse_all(&ADAPTER, &[&fixture("reasoning.sse")]);
        assert_eq!(deltas(&events), [DeltaKind::Text]);
        assert_eq!(total_usage(&events).reasoning_tokens, Some(192));
    }

    #[test]
    fn reasoning_content_is_thinking() {
        let events = parse_all(&ADAPTER, &[&fixture("reasoning_content.sse")]);
        assert_eq!(deltas(&events), [DeltaKind::Thinking, DeltaKind::Text]);
        assert_eq!(
            events[0],
            StreamEvent::ResponseModel("Qwen/Qwen3-8B".into())
        );
        assert_eq!(total_usage(&events).input_tokens_total, Some(18));
    }

    #[test]
    fn tool_calls_are_tool_input() {
        let events = parse_all(&ADAPTER, &[&fixture("tool_use.sse")]);
        assert_eq!(deltas(&events), [DeltaKind::ToolInput]);
        assert!(events.contains(&StreamEvent::Stop("tool_calls".into())));
    }

    #[test]
    fn cache_hit_reads_cached_tokens() {
        let u = total_usage(&parse_all(&ADAPTER, &[&fixture("cache_hit.sse")]));
        assert_eq!(u.input_tokens_total, Some(7040));
        assert_eq!(u.cache_read_tokens, Some(6912));
        assert_eq!(u.cache_write_tokens, Some(0));
    }

    #[test]
    fn non_stream_json() {
        let events = ADAPTER.parse_response(&fixture("text.json"));
        assert!(matches!(&events[0], StreamEvent::ResponseModel(_)));
        assert!(events.contains(&StreamEvent::Stop("stop".into())));
        assert_eq!(total_usage(&events).output_tokens, Some(74));
    }

    #[test]
    fn real_error_bodies() {
        for (name, kind) in [
            ("error_401.json", "invalid_request_error"),
            ("insufficient_quota.json", "insufficient_quota"),
        ] {
            let events = ADAPTER.parse_response(&fixture(name));
            assert!(
                matches!(&events[..], [StreamEvent::Error { kind: k, .. }] if k == kind),
                "{name}: {events:?}"
            );
        }
    }

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(format!(
            "{}/tests/fixtures/openai/{name}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap()
    }

    proptest! {
        #[test]
        fn random_splits_match(cuts in proptest::collection::vec(any::<usize>(), 0..30)) {
            for (name, fixture) in sse_fixtures("openai") {
                let whole = parse_all(&ADAPTER, &[&fixture]);
                let chunks = split_at(&fixture, &cuts);
                let refs: Vec<&[u8]> = chunks.iter().map(Vec::as_slice).collect();
                prop_assert_eq!(parse_all(&ADAPTER, &refs), whole, "{}", name);
            }
        }
    }
}
