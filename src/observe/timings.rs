//! Request timings, measured with `tokio::time::Instant` so tests can pause time.

use std::time::Duration;

use tokio::time::Instant;

use crate::providers::DeltaKind;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Timings {
    /// Upstream response headers received.
    pub ttfb: Option<Duration>,
    /// First content delta of any kind.
    pub ttft: Option<Duration>,
    /// First visible text delta.
    pub ttft_text: Option<Duration>,
    /// Until the last byte, failure or cancellation.
    pub duration: Duration,
    pub chunk_count: u64,
    /// Chunks are not tokens: an inter-token latency proxy.
    pub chunk_gap_mean: Option<Duration>,
    pub chunk_gap_max: Option<Duration>,
    /// `output_tokens / (end - first token)`.
    pub output_tps: Option<f64>,
}

#[derive(Debug)]
pub struct Clock {
    start: Instant,
    ttfb: Option<Duration>,
    first_token: Option<Instant>,
    first_text: Option<Instant>,
    last_chunk: Option<Instant>,
    chunk_count: u64,
    gap_sum: Duration,
    gap_max: Duration,
}

impl Clock {
    #[must_use]
    pub fn new(start: Instant) -> Self {
        Self {
            start,
            ttfb: None,
            first_token: None,
            first_text: None,
            last_chunk: None,
            chunk_count: 0,
            gap_sum: Duration::ZERO,
            gap_max: Duration::ZERO,
        }
    }

    pub fn headers(&mut self, now: Instant) {
        self.ttfb.get_or_insert(now - self.start);
    }

    pub fn chunk(&mut self, now: Instant) {
        if let Some(last) = self.last_chunk {
            let gap = now - last;
            self.gap_sum += gap;
            self.gap_max = self.gap_max.max(gap);
        }
        self.last_chunk = Some(now);
        self.chunk_count += 1;
    }

    pub fn delta(&mut self, kind: DeltaKind, now: Instant) {
        self.first_token.get_or_insert(now);
        if kind == DeltaKind::Text {
            self.first_text.get_or_insert(now);
        }
    }

    #[must_use]
    pub fn finish(&self, now: Instant, output_tokens: Option<u64>) -> Timings {
        let gaps = self.chunk_count.saturating_sub(1);
        let (gap_mean, gap_max) = match u32::try_from(gaps) {
            Ok(n) if n > 0 => (Some(self.gap_sum / n), Some(self.gap_max)),
            _ => (None, None),
        };
        #[allow(clippy::cast_precision_loss)]
        let output_tps = match (output_tokens, self.first_token) {
            (Some(tokens), Some(first)) if now > first => {
                Some(tokens as f64 / (now - first).as_secs_f64())
            }
            _ => None,
        };
        Timings {
            ttfb: self.ttfb,
            ttft: self.first_token.map(|t| t - self.start),
            ttft_text: self.first_text.map(|t| t - self.start),
            duration: now - self.start,
            chunk_count: self.chunk_count,
            chunk_gap_mean: gap_mean,
            chunk_gap_max: gap_max,
            output_tps,
        }
    }
}
