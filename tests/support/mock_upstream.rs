//! Scripted upstream for integration tests. It replays a response chunk by
//! chunk with per-chunk delays, and records what it received. It notices
//! when the gateway drops the response before the last chunk: that means a
//! client disconnect propagated upstream.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, Method, StatusCode, Uri};
use axum::response::Response;
use futures_util::stream;
use http_body_util::BodyExt;
use tokio::sync::Notify;
use tokio::time::Instant;

/// What the mock answers with.
#[derive(Clone, Debug)]
pub struct Script {
    pub status: StatusCode,
    pub headers: Vec<(String, String)>,
    /// `(delay before this chunk, chunk)`.
    pub chunks: Vec<(Duration, Bytes)>,
    /// Delay before the status line and headers are sent.
    pub header_delay: Duration,
}

impl Script {
    /// An SSE response, one chunk per event, `gap` between events.
    pub fn sse(fixture: &[u8], gap: Duration) -> Self {
        let chunks = split_events(fixture)
            .into_iter()
            .enumerate()
            .map(|(i, event)| (if i == 0 { Duration::ZERO } else { gap }, event))
            .collect();
        Self {
            status: StatusCode::OK,
            headers: vec![
                (
                    "content-type".into(),
                    "text/event-stream; charset=utf-8".into(),
                ),
                ("cache-control".into(), "no-cache".into()),
                ("request-id".into(), "req_mock_1".into()),
                ("anthropic-ratelimit-requests-remaining".into(), "49".into()),
            ],
            chunks,
            header_delay: Duration::ZERO,
        }
    }

    /// A single-chunk response with an explicit `content-length`.
    pub fn json(status: StatusCode, body: &[u8]) -> Self {
        Self {
            status,
            headers: vec![
                ("content-type".into(), "application/json".into()),
                ("content-length".into(), body.len().to_string()),
            ],
            chunks: vec![(Duration::ZERO, Bytes::copy_from_slice(body))],
            header_delay: Duration::ZERO,
        }
    }

    /// Explicit chunks, e.g. two chunks with a long gap.
    pub fn chunks(chunks: Vec<(Duration, &'static str)>) -> Self {
        Self {
            status: StatusCode::OK,
            headers: vec![("content-type".into(), "text/event-stream".into())],
            chunks: chunks
                .into_iter()
                .map(|(d, c)| (d, Bytes::from_static(c.as_bytes())))
                .collect(),
            header_delay: Duration::ZERO,
        }
    }

    #[must_use]
    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        self.headers.push((name.to_owned(), value.to_owned()));
        self
    }

    /// Replace headers with a `*.headers` fixture (status line skipped).
    #[must_use]
    pub fn with_fixture_headers(mut self, raw: &str) -> Self {
        self.headers = raw
            .lines()
            .skip(1)
            .filter_map(|l| l.split_once(':'))
            .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
            .filter(|(k, _)| k != "content-length")
            .collect();
        self
    }

    pub fn body(&self) -> Vec<u8> {
        self.chunks
            .iter()
            .flat_map(|(_, c)| c.iter().copied())
            .collect()
    }
}

/// Split SSE text into events, each keeping its trailing blank line.
fn split_events(text: &[u8]) -> Vec<Bytes> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i + 1 < text.len() {
        if text[i] == b'\n' && text[i + 1] == b'\n' {
            out.push(Bytes::copy_from_slice(&text[start..i + 2]));
            start = i + 2;
            i += 2;
        } else {
            i += 1;
        }
    }
    if start < text.len() {
        out.push(Bytes::copy_from_slice(&text[start..]));
    }
    out
}

#[derive(Clone, Debug)]
pub struct Recorded {
    pub method: Method,
    pub uri: Uri,
    pub headers: HeaderMap,
    pub body: Bytes,
}

#[derive(Default)]
struct Log {
    requests: Vec<Recorded>,
    sent_at: Vec<Instant>,
    disconnected_at: Option<Instant>,
}

struct Shared {
    script: Mutex<Script>,
    /// Answered first, one per request, before `script`.
    queued: Mutex<std::collections::VecDeque<Script>>,
    log: Mutex<Log>,
    disconnected: Notify,
}

pub struct MockUpstream {
    pub addr: SocketAddr,
    shared: Arc<Shared>,
}

impl MockUpstream {
    pub async fn start(script: Script) -> Self {
        Self::start_seq(vec![script]).await
    }

    /// Answers each request with the next script; the last one repeats.
    pub async fn start_seq(mut scripts: Vec<Script>) -> Self {
        let last = scripts.pop().expect("at least one script");
        let shared = Arc::new(Shared {
            script: Mutex::new(last),
            queued: Mutex::new(scripts.into()),
            log: Mutex::new(Log::default()),
            disconnected: Notify::new(),
        });
        let app = Router::new().fallback(handle).with_state(shared.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self { addr, shared }
    }

    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.shared.log.lock().unwrap().requests.clone()
    }

    pub fn last_request(&self) -> Recorded {
        self.requests().pop().expect("mock received no request")
    }

    /// When each chunk was handed to hyper, by index.
    pub fn sent_at(&self) -> Vec<Instant> {
        self.shared.log.lock().unwrap().sent_at.clone()
    }

    /// Wait until the response body is dropped before its last chunk.
    pub async fn wait_disconnect(&self, limit: Duration) -> Option<Instant> {
        let _ = tokio::time::timeout(limit, self.shared.disconnected.notified()).await;
        self.shared.log.lock().unwrap().disconnected_at
    }
}

/// Records a disconnect when the body stream is dropped unfinished.
struct DropGuard {
    shared: Arc<Shared>,
    finished: bool,
}

impl Drop for DropGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.shared.log.lock().unwrap().disconnected_at = Some(Instant::now());
            self.shared.disconnected.notify_one();
        }
    }
}

async fn handle(State(shared): State<Arc<Shared>>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let body = body.collect().await.unwrap().to_bytes();
    shared.log.lock().unwrap().requests.push(Recorded {
        method: parts.method,
        uri: parts.uri,
        headers: parts.headers,
        body,
    });

    let queued = shared.queued.lock().unwrap().pop_front();
    let script = queued.unwrap_or_else(|| shared.script.lock().unwrap().clone());
    tokio::time::sleep(script.header_delay).await;

    let guard = DropGuard {
        shared: shared.clone(),
        finished: false,
    };
    let chunks = script.chunks;
    let body = stream::unfold((0usize, guard), move |(i, mut guard)| {
        let chunks = chunks.clone();
        async move {
            let (delay, chunk) = chunks.get(i)?.clone();
            tokio::time::sleep(delay).await;
            guard
                .shared
                .log
                .lock()
                .unwrap()
                .sent_at
                .push(Instant::now());
            if i + 1 == chunks.len() {
                guard.finished = true;
            }
            Some((Ok::<_, std::io::Error>(chunk), (i + 1, guard)))
        }
    });

    let mut resp = Response::builder().status(script.status);
    for (name, value) in &script.headers {
        resp = resp.header(name, value);
    }
    resp.body(Body::from_stream(body)).unwrap()
}

/// An address nothing listens on.
pub fn unreachable_addr() -> SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap()
}
