//! Renders an evaluator's `input` template from captured content, cut to
//! fit the judge's token budget.

use std::collections::BTreeMap;

use anyhow::Result;
use minijinja::Environment;
use serde_json::Value;

use crate::config::eval::Question;
use crate::providers::Output;
use crate::store::TraceContent;

/// Conservative: real text averages more chars per token.
pub const CHARS_PER_TOKEN: usize = 3;

const CUT: &str = "\n[…truncated…]\n";
const SEPARATOR: &str = "\n\n";

/// Template variables.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Vars {
    pub last_user_message: String,
    pub output_text: String,
    pub system: String,
    /// One entry per message, oldest first, as `role: text`.
    pub conversation: Vec<String>,
    /// One line per call in the response: `name {input}`.
    pub tool_calls: String,
    /// Visible thinking; empty when the provider hides it (Claude Code).
    pub thinking: String,
    /// Raw provider value, e.g. `end_turn`, `tool_use`, `stop`.
    pub stop_reason: String,
}

/// Names in the template; `conversation` is cut differently.
const NAMES: [&str; 7] = [
    "last_user_message",
    "output_text",
    "system",
    "conversation",
    "tool_calls",
    "thinking",
    "stop_reason",
];
const CONVERSATION: usize = 3;

impl Vars {
    /// `output` is the response re-extracted from the stored raw bytes;
    /// `None` when that failed or no response arrived.
    #[must_use]
    pub fn from_content(
        c: &TraceContent,
        output: Option<&Output>,
        stop_reason: Option<&str>,
    ) -> Self {
        let req = &c.request;
        let system = match &req.system {
            Some(s) => text_of(s),
            // OpenAI keeps system prompts in messages.
            None => req
                .messages
                .iter()
                .filter(|m| matches!(role(m), "system" | "developer"))
                .map(|m| m.get("content").map(text_of).unwrap_or_default())
                .collect::<Vec<_>>()
                .join(SEPARATOR),
        };
        let last_user = c
            .last_user_text
            .clone()
            .or_else(|| req.last_user_text())
            .unwrap_or_default();
        let stripped = strip_wrappers(&last_user);
        Self {
            // A message of only wrappers stays as is rather than going blank.
            last_user_message: if stripped.is_empty() {
                last_user
            } else {
                stripped
            },
            output_text: c.output_text.clone().unwrap_or_default(),
            system,
            conversation: req.messages.iter().map(message).collect(),
            tool_calls: output
                .map(|o| {
                    o.tool_calls
                        .iter()
                        .map(|t| format!("{} {}", t.name, t.input))
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default(),
            thinking: output.map(|o| o.thinking.clone()).unwrap_or_default(),
            stop_reason: stop_reason.unwrap_or_default().to_owned(),
        }
    }

    fn conversation_from(&self, from: usize) -> String {
        self.conversation[from.min(self.conversation.len())..].join(SEPARATOR)
    }

    /// In [`NAMES`] order, with `conversation` from message `from` on.
    fn values(&self, from: usize) -> [String; 7] {
        [
            self.last_user_message.clone(),
            self.output_text.clone(),
            self.system.clone(),
            self.conversation_from(from),
            self.tool_calls.clone(),
            self.thinking.clone(),
            self.stop_reason.clone(),
        ]
    }
}

fn role(m: &Value) -> &str {
    m.get("role").and_then(Value::as_str).unwrap_or("unknown")
}

/// Elements Claude Code adds to user messages: context it injects, not what
/// the user wrote.
const WRAPPERS: [&str; 6] = [
    "system-reminder",
    "local-command-caveat",
    "command-name",
    "command-message",
    "command-args",
    "local-command-stdout",
];

/// Removes [`WRAPPERS`] elements and trims.
fn strip_wrappers(s: &str) -> String {
    let mut out = s.to_owned();
    for tag in WRAPPERS {
        let (open, close) = (format!("<{tag}>"), format!("</{tag}>"));
        let mut from = 0;
        while let Some(start) = out[from..].find(&open).map(|i| from + i) {
            let Some(end) = out[start..].find(&close).map(|i| start + i + close.len()) else {
                break;
            };
            out.replace_range(start..end, "");
            from = start;
        }
    }
    out.trim().to_owned()
}

/// `role: text`, with tool calls and results inline.
fn message(m: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(c) = m.get("content") {
        let mut text = text_of(c);
        if role(m) == "user" {
            text = strip_wrappers(&text);
        }
        if !text.is_empty() {
            parts.push(text);
        }
    }
    // OpenAI assistant tool calls.
    for call in m
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let f = &call["function"];
        parts.push(format!(
            "[tool_call {} {}]",
            f["name"].as_str().unwrap_or_default(),
            f["arguments"].as_str().unwrap_or_default()
        ));
    }
    format!("{}: {}", role(m), parts.join("\n"))
}

/// Text of a string or a content-block array (Anthropic or `OpenAI`).
fn text_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(block)
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn block(b: &Value) -> Option<String> {
    let kind = b.get("type").and_then(Value::as_str).unwrap_or_default();
    match kind {
        "text" => b.get("text").and_then(Value::as_str).map(str::to_owned),
        // Hidden in Claude Code traffic, and not what is being judged.
        "thinking" | "redacted_thinking" => None,
        "tool_use" => Some(format!(
            "[tool_use {} {}]",
            b["name"].as_str().unwrap_or_default(),
            b["input"]
        )),
        "tool_result" => Some(format!(
            "[tool_result {}]",
            b.get("content").map(text_of).unwrap_or_default()
        )),
        other => Some(format!("[{other}]")),
    }
}

/// Chars available for the input: the token budget minus the question.
#[must_use]
pub fn budget_chars(max_input_tokens: usize, name: &str, question: &Question) -> usize {
    let question_chars = serde_json::to_string(question).map_or(0, |q| q.len()) + name.len();
    let question_tokens = question_chars.div_ceil(CHARS_PER_TOKEN);
    max_input_tokens.saturating_sub(question_tokens) * CHARS_PER_TOKEN
}

/// Returns the input and whether any variable was cut. Over budget, the
/// oldest conversation messages go first; then the longest variables are
/// cut in the middle, keeping head and tail, until the render fits.
///
/// # Errors
///
/// Template syntax or render error.
pub fn render(template: &str, vars: &Vars, budget: usize) -> Result<(String, bool)> {
    let env = Environment::new();
    let render = |values: &[String; 7]| -> Result<String> {
        let ctx: BTreeMap<&str, &str> = NAMES
            .iter()
            .zip(values)
            .map(|(n, v)| (*n, v.as_str()))
            .collect();
        Ok(env.render_str(template, ctx)?)
    };
    let mut values = vars.values(0);
    let out = render(&values)?;
    if chars(&out) <= budget {
        return Ok((out, false));
    }

    // Each variable's length counts once per use in the template.
    let markers: [String; 7] = std::array::from_fn(|i| format!("\u{1}{i}\u{1}"));
    let probe = render(&markers)?;
    let uses = markers
        .each_ref()
        .map(|m| probe.matches(m.as_str()).count());
    let overhead = chars(&probe)
        - markers
            .iter()
            .zip(uses)
            .map(|(m, n)| chars(m) * n)
            .sum::<usize>();
    let available = budget.saturating_sub(overhead);
    let cost = |lens: &[usize; 7]| lens.iter().zip(uses).map(|(l, n)| l * n).sum::<usize>();

    let mut lens = values.each_ref().map(|v| chars(v));
    // Dropping a message (never the last) removes it and one separator.
    let mut dropped = 0;
    while cost(&lens) > available && dropped + 1 < vars.conversation.len() {
        lens[CONVERSATION] -= chars(&vars.conversation[dropped]) + chars(SEPARATOR);
        dropped += 1;
    }
    if dropped > 0 {
        values[CONVERSATION] = vars.conversation_from(dropped);
        debug_assert_eq!(lens[CONVERSATION], chars(&values[CONVERSATION]));
    }
    // Largest cap that fits, found by bisection.
    let (mut lo, mut hi) = (0, lens.into_iter().max().unwrap_or(0));
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if cost(&lens.map(|l| l.min(mid))) <= available {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let cut = values.map(|v| cut_middle(&v, lo));
    Ok((render(&cut)?, true))
}

fn chars(s: &str) -> usize {
    s.chars().count()
}

/// At most `max` chars: head, marker, tail.
fn cut_middle(s: &str, max: usize) -> String {
    let n = chars(s);
    if n <= max {
        return s.to_owned();
    }
    let keep = max.saturating_sub(chars(CUT));
    if keep == 0 {
        return s.chars().take(max).collect();
    }
    let head = keep.div_ceil(2);
    let tail = keep - head;
    let mut out: String = s.chars().take(head).collect();
    out.push_str(CUT);
    out.extend(s.chars().skip(n - tail));
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::providers::RequestParts;

    const T: &str = "<user>{{ last_user_message }}</user>\n<response>{{ output_text }}</response>";

    fn vars(output: &str) -> Vars {
        Vars {
            last_user_message: "question?".into(),
            output_text: output.into(),
            ..Vars::default()
        }
    }

    #[test]
    fn under_budget_is_unchanged() {
        let (out, cut) = render(T, &vars("answer"), 1000).unwrap();
        assert_eq!(out, "<user>question?</user>\n<response>answer</response>");
        assert!(!cut);
    }

    #[test]
    fn over_budget_keeps_head_and_tail() {
        let output = format!("HEAD{}TAIL", "x".repeat(10_000));
        let (out, cut) = render(T, &vars(&output), 500).unwrap();
        assert!(cut);
        assert!(chars(&out) <= 500, "{}", chars(&out));
        assert!(out.contains("<response>HEAD"));
        assert!(out.contains("TAIL</response>"));
        assert!(out.contains("[…truncated…]"));
        // The short variable is untouched.
        assert!(out.contains("<user>question?</user>"));
    }

    #[test]
    fn conversation_drops_oldest_first() {
        let v = Vars {
            conversation: vec![
                format!("user: OLDEST {}", "a".repeat(400)),
                format!("assistant: {}", "b".repeat(400)),
                "user: newest".into(),
            ],
            ..Vars::default()
        };
        let (out, cut) = render("{{ conversation }}", &v, 450).unwrap();
        assert!(cut);
        assert!(!out.contains("OLDEST"));
        assert!(out.ends_with("user: newest"), "{out}");
    }

    #[test]
    fn long_conversation_is_cut_in_one_pass() {
        // About a 1M-token session: re-joining per dropped message would be
        // quadratic here.
        let conversation: Vec<String> = (0..2000)
            .map(|i| format!("user: {i:04} {}", "c".repeat(2000)))
            .collect();
        let v = Vars {
            conversation,
            ..Vars::default()
        };
        let (out, cut) = render("{{ conversation }}", &v, 96_000).unwrap();
        assert!(cut);
        assert!(chars(&out) <= 96_000, "{}", chars(&out));
        assert!(out.contains("user: 1999 "));
        assert!(!out.contains("user: 0000 "));
    }

    #[test]
    fn variable_used_twice_counts_twice() {
        let (out, _) = render(
            "{{ output_text }}|{{ output_text }}",
            &vars(&"z".repeat(1000)),
            300,
        )
        .unwrap();
        assert!(chars(&out) <= 300, "{}", chars(&out));
    }

    #[test]
    fn multibyte_text_is_cut_on_char_boundaries() {
        let (out, cut) = render("{{ output_text }}", &vars(&"é😀".repeat(500)), 100).unwrap();
        assert!(cut);
        assert!(chars(&out) <= 100);
    }

    #[test]
    fn vars_from_anthropic_content() {
        let c = TraceContent {
            request: RequestParts {
                system: Some(json!([{"type": "text", "text": "be nice"}])),
                tools: None,
                messages: vec![
                    json!({"role": "user", "content": "list files"}),
                    json!({"role": "assistant", "content": [
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "tool_use", "id": "t1", "name": "ls", "input": {"path": "."}},
                    ]}),
                    json!({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "a.rs"},
                    ]}),
                ],
                params: serde_json::Map::new(),
            },
            raw_response: Vec::new(),
            output_text: Some("There is a.rs".into()),
            last_user_text: Some("list files".into()),
            truncated: false,
        };
        let v = Vars::from_content(&c, None, None);
        assert_eq!(v.system, "be nice");
        assert_eq!(v.last_user_message, "list files");
        assert_eq!(v.output_text, "There is a.rs");
        assert_eq!(
            v.conversation,
            [
                "user: list files",
                r#"assistant: [tool_use ls {"path":"."}]"#,
                "user: [tool_result a.rs]",
            ]
        );
    }

    #[test]
    fn claude_code_wrappers_are_stripped() {
        let prompt = "<system-reminder>\nCLAUDE.md says hi\n</system-reminder>\n\
            <local-command-caveat>Caveat</local-command-caveat>\n\
            <command-name>/clear</command-name>\n<command-args></command-args>\n\
            <system-reminder>git status</system-reminder>\n\nReview phase 4.";
        let c = TraceContent {
            request: RequestParts {
                messages: vec![
                    json!({"role": "user", "content": [
                        {"type": "text", "text": "<system-reminder>ctx</system-reminder>"},
                        {"type": "text", "text": "fix the bug"},
                    ]}),
                    json!({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "t1",
                         "content": "ok\n<system-reminder>todo</system-reminder>"},
                    ]}),
                    json!({"role": "user", "content": "<system-reminder>only</system-reminder>"}),
                ],
                ..RequestParts::default()
            },
            raw_response: Vec::new(),
            output_text: None,
            last_user_text: Some(prompt.into()),
            truncated: false,
        };
        let v = Vars::from_content(&c, None, None);
        assert_eq!(v.last_user_message, "Review phase 4.");
        assert_eq!(
            v.conversation,
            ["user: fix the bug", "user: [tool_result ok\n]", "user: "]
        );

        let only = TraceContent {
            last_user_text: Some("<system-reminder>x</system-reminder>".into()),
            ..c
        };
        assert_eq!(
            Vars::from_content(&only, None, None).last_user_message,
            "<system-reminder>x</system-reminder>"
        );
        assert_eq!(
            strip_wrappers("a <system-reminder>unclosed"),
            "a <system-reminder>unclosed"
        );
    }

    #[test]
    fn vars_from_openai_content() {
        let c = TraceContent {
            request: RequestParts {
                system: None,
                tools: None,
                messages: vec![
                    json!({"role": "system", "content": "be terse"}),
                    json!({"role": "user", "content": [{"type": "text", "text": "hi"}]}),
                    json!({"role": "assistant", "content": null, "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                    ]}),
                    json!({"role": "tool", "tool_call_id": "c1", "content": "42"}),
                ],
                params: serde_json::Map::new(),
            },
            raw_response: Vec::new(),
            output_text: None,
            last_user_text: None,
            truncated: false,
        };
        let v = Vars::from_content(&c, None, None);
        assert_eq!(v.system, "be terse");
        assert_eq!(v.last_user_message, "hi");
        assert_eq!(v.conversation[2], "assistant: [tool_call f {}]");
        assert_eq!(v.conversation[3], "tool: 42");
    }

    #[test]
    fn tool_calls_thinking_and_stop_reason_from_output() {
        let raw = include_bytes!("../../tests/fixtures/anthropic/cc_tool_use.sse");
        let out = crate::providers::ProviderKind::Anthropic
            .extract_output(raw, true)
            .unwrap();
        assert!(!out.tool_calls.is_empty());
        let c = TraceContent {
            request: RequestParts::default(),
            raw_response: Vec::new(),
            output_text: None,
            last_user_text: None,
            truncated: false,
        };
        let v = Vars::from_content(&c, Some(&out), Some("tool_use"));
        let first = &out.tool_calls[0];
        assert!(
            v.tool_calls
                .starts_with(&format!("{} {}", first.name, first.input)),
            "{}",
            v.tool_calls
        );
        assert_eq!(v.tool_calls.lines().count(), out.tool_calls.len());
        assert_eq!(v.stop_reason, "tool_use");
        let (state, cut) = render("{{ stop_reason }}: {{ tool_calls }}", &v, 100_000).unwrap();
        assert!(!cut);
        assert!(
            state.starts_with(&format!("tool_use: {}", first.name)),
            "{state}"
        );
    }

    #[test]
    fn openai_tool_calls_render_too() {
        let raw = include_bytes!("../../tests/fixtures/openai/tool_use.sse");
        let out = crate::providers::ProviderKind::Openai
            .extract_output(raw, true)
            .unwrap();
        let c = TraceContent {
            request: RequestParts::default(),
            raw_response: Vec::new(),
            output_text: None,
            last_user_text: None,
            truncated: false,
        };
        let v = Vars::from_content(&c, Some(&out), Some("tool_calls"));
        assert_eq!(v.tool_calls.lines().count(), out.tool_calls.len());
        assert!(v.tool_calls.starts_with(&out.tool_calls[0].name));
    }

    #[test]
    fn tool_calls_are_cut_like_other_variables() {
        let v = Vars {
            tool_calls: format!("Bash {}", "x".repeat(5000)),
            ..Vars::default()
        };
        let (out, cut) = render("{{ tool_calls }}", &v, 200).unwrap();
        assert!(cut);
        assert!(chars(&out) <= 200);
        assert!(out.starts_with("Bash "));
    }

    #[test]
    fn budget_subtracts_the_question() {
        let q = Question::YesNo {
            instructions: "x".into(),
            criteria: None,
        };
        let b = budget_chars(1000, "q", &q);
        assert!(b < 3000 && b > 2900, "{b}");
    }
}
