//! Incremental Server-Sent Events parser (WHATWG `text/event-stream`).
//! Chunks may split lines and events anywhere, including between `\r` and `\n`.

/// One dispatched event. `data` lines are joined with `\n`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

#[derive(Debug, Default)]
pub struct SseParser {
    line: Vec<u8>,
    /// Last chunk ended in `\r`; a leading `\n` in the next one belongs to it.
    after_cr: bool,
    event: Option<String>,
    data: String,
    has_data: bool,
}

impl SseParser {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, mut chunk: &[u8], out: &mut Vec<SseEvent>) {
        if chunk.is_empty() {
            return;
        }
        if self.after_cr {
            self.after_cr = false;
            if let Some(rest) = chunk.strip_prefix(b"\n") {
                chunk = rest;
            }
        }
        while let Some(i) = chunk.iter().position(|&b| b == b'\n' || b == b'\r') {
            self.line.extend_from_slice(&chunk[..i]);
            let line = std::mem::take(&mut self.line);
            self.process_line(&line, out);
            self.line = line;
            self.line.clear();

            let cr = chunk[i] == b'\r';
            chunk = &chunk[i + 1..];
            if cr {
                match chunk.first() {
                    Some(b'\n') => chunk = &chunk[1..],
                    Some(_) => {}
                    None => self.after_cr = true,
                }
            }
        }
        self.line.extend_from_slice(chunk);
    }

    /// Dispatch a final event the stream did not terminate with a blank line.
    pub fn finish(&mut self, out: &mut Vec<SseEvent>) {
        if !self.line.is_empty() {
            let line = std::mem::take(&mut self.line);
            self.process_line(&line, out);
        }
        self.dispatch(out);
    }

    fn process_line(&mut self, line: &[u8], out: &mut Vec<SseEvent>) {
        if line.is_empty() {
            self.dispatch(out);
            return;
        }
        if line[0] == b':' {
            return;
        }
        let (field, value) = match line.iter().position(|&b| b == b':') {
            Some(i) => {
                let v = &line[i + 1..];
                (&line[..i], v.strip_prefix(b" ").unwrap_or(v))
            }
            None => (line, &b""[..]),
        };
        match field {
            b"event" => self.event = Some(String::from_utf8_lossy(value).into_owned()),
            b"data" => {
                if self.has_data {
                    self.data.push('\n');
                }
                self.data.push_str(&String::from_utf8_lossy(value));
                self.has_data = true;
            }
            _ => {}
        }
    }

    fn dispatch(&mut self, out: &mut Vec<SseEvent>) {
        let event = self.event.take();
        if self.has_data {
            self.has_data = false;
            out.push(SseEvent {
                event,
                data: std::mem::take(&mut self.data),
            });
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn parse_chunks(chunks: &[&[u8]]) -> Vec<SseEvent> {
        let mut p = SseParser::new();
        let mut out = Vec::new();
        for c in chunks {
            p.push(c, &mut out);
        }
        p.finish(&mut out);
        out
    }

    fn ev(event: Option<&str>, data: &str) -> SseEvent {
        SseEvent {
            event: event.map(str::to_owned),
            data: data.to_owned(),
        }
    }

    #[test]
    fn parses_events_and_comments() {
        let out = parse_chunks(&[b": keepalive\n\nevent: a\ndata: {\"x\":1}\n\ndata:no-space\n\n"]);
        assert_eq!(out, [ev(Some("a"), "{\"x\":1}"), ev(None, "no-space")]);
    }

    #[test]
    fn joins_multi_line_data() {
        let out = parse_chunks(&[b"data: a\ndata: b\ndata\n\n"]);
        assert_eq!(out, [ev(None, "a\nb\n")]);
    }

    #[test]
    fn handles_crlf_and_cr() {
        let out = parse_chunks(&[b"data: a\r\n\r\ndata: b\r\rdata: c\n\n"]);
        assert_eq!(out, [ev(None, "a"), ev(None, "b"), ev(None, "c")]);
    }

    #[test]
    fn crlf_split_across_chunks() {
        let out = parse_chunks(&[b"data: a\r", b"\n\r", b"\ndata: b\r\n\r\n"]);
        assert_eq!(out, [ev(None, "a"), ev(None, "b")]);
    }

    #[test]
    fn event_without_data_is_dropped() {
        let out = parse_chunks(&[b"event: ping\n\ndata: x\n\n"]);
        assert_eq!(out, [ev(None, "x")]);
    }

    #[test]
    fn finish_flushes_unterminated_event() {
        let out = parse_chunks(&[b"data: a\n\ndata: [DONE]"]);
        assert_eq!(out, [ev(None, "a"), ev(None, "[DONE]")]);
    }

    proptest! {
        #[test]
        fn any_split_gives_same_events(
            cuts in proptest::collection::vec(0usize..200, 0..20),
            crlf in any::<bool>(),
        ) {
            let text: &[u8] = if crlf {
                b"event: a\r\ndata: 1\r\n\r\n: c\r\ndata: 2\r\ndata: 3\r\n\r\nevent: b\r\ndata: {\"k\":\"v\"}\r\n\r\n"
            } else {
                b"event: a\ndata: 1\n\n: c\ndata: 2\ndata: 3\n\nevent: b\ndata: {\"k\":\"v\"}\n\n"
            };
            let mut cuts: Vec<usize> = cuts.into_iter().map(|c| c % (text.len() + 1)).collect();
            cuts.sort_unstable();
            let mut chunks = Vec::new();
            let mut prev = 0;
            for c in cuts {
                chunks.push(&text[prev..c]);
                prev = c;
            }
            chunks.push(&text[prev..]);
            prop_assert_eq!(
                parse_chunks(&chunks),
                vec![ev(Some("a"), "1"), ev(None, "2\n3"), ev(Some("b"), "{\"k\":\"v\"}")]
            );
        }
    }
}
