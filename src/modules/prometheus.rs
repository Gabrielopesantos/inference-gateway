//! Prometheus metrics, served on the admin listener at `/metrics`.

use std::sync::Arc;

use ::prometheus::core::Collector;
use ::prometheus::{
    CounterVec, Encoder as _, GaugeVec, HistogramOpts, HistogramVec, IntCounterVec, Opts, Registry,
    TextEncoder,
};
use anyhow::Result;
use async_trait::async_trait;
use tracing::warn;

use super::{BuildCtx, CompletedExchange, DropCounters, Module, Phase, Scrape};
use crate::config::PrometheusFile;
use crate::observe::record::RequestRecord;

const REQUEST_LABELS: &[&str] = &["upstream", "provider", "model", "outcome", "streaming"];
const MODEL_LABELS: &[&str] = &["upstream", "provider", "model"];

/// Clones share the same metrics.
#[derive(Clone)]
pub struct Prometheus {
    registry: Registry,
    drops: Arc<DropCounters>,
    dropped: IntCounterVec,
    requests: IntCounterVec,
    tokens: IntCounterVec,
    cost: CounterVec,
    ttfb: HistogramVec,
    ttft: HistogramVec,
    duration: HistogramVec,
    output_tps: HistogramVec,
    chunk_gap: HistogramVec,
    ratelimit_remaining: GaugeVec,
    window_utilization: GaugeVec,
}

impl Prometheus {
    /// # Errors
    ///
    /// Metric registration fails (duplicate or invalid name).
    pub fn new(_cfg: &PrometheusFile, ctx: &BuildCtx) -> Result<Self> {
        let registry = Registry::new();
        let counter = |name: &str, help: &str, labels: &[&str]| {
            register(
                &registry,
                IntCounterVec::new(Opts::new(name, help), labels)?,
            )
        };
        let gauge = |name: &str, help: &str, labels: &[&str]| {
            register(&registry, GaugeVec::new(Opts::new(name, help), labels)?)
        };
        let histogram = |name: &str, help: &str, labels: &[&str], buckets: &[f64]| {
            let opts = HistogramOpts::new(name, help).buckets(buckets.to_vec());
            register(&registry, HistogramVec::new(opts, labels)?)
        };
        let latency = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];

        Ok(Self {
            drops: ctx.drops.clone(),
            dropped: counter(
                "gateway_module_dropped_total",
                "Records an export module dropped because its queue was full.",
                &["module"],
            )?,
            requests: counter(
                "gateway_requests_total",
                "Proxied requests.",
                REQUEST_LABELS,
            )?,
            tokens: counter(
                "gateway_tokens_total",
                "Tokens by kind; input includes cache reads and writes.",
                &["upstream", "provider", "model", "kind"],
            )?,
            cost: register(
                &registry,
                CounterVec::new(
                    Opts::new("gateway_cost_usd_total", "Estimated spend in USD."),
                    MODEL_LABELS,
                )?,
            )?,
            ttfb: histogram(
                "gateway_ttfb_seconds",
                "Time to upstream response headers.",
                MODEL_LABELS,
                &latency,
            )?,
            ttft: histogram(
                "gateway_ttft_seconds",
                "Time to first content delta of any kind.",
                MODEL_LABELS,
                &latency,
            )?,
            duration: histogram(
                "gateway_request_duration_seconds",
                "Time to last byte, failure or cancellation.",
                REQUEST_LABELS,
                &[
                    0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0,
                ],
            )?,
            output_tps: histogram(
                "gateway_output_tokens_per_second",
                "Decode throughput after the first token.",
                MODEL_LABELS,
                &[5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0],
            )?,
            chunk_gap: histogram(
                "gateway_chunk_gap_seconds",
                "Mean gap between response chunks per request.",
                MODEL_LABELS,
                &[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
            )?,
            ratelimit_remaining: gauge(
                "gateway_upstream_ratelimit_remaining",
                "Remaining upstream rate limit from the last response.",
                &["upstream", "model", "kind"],
            )?,
            window_utilization: gauge(
                "gateway_subscription_window_utilization",
                "Used fraction of a subscription window (0..1) from the last response.",
                &["upstream", "window"],
            )?,
            registry,
        })
    }

    #[allow(clippy::cast_precision_loss)]
    fn observe(&self, r: &RequestRecord) {
        let model = r.model().unwrap_or("unknown");
        let provider = r.provider.as_str();
        let streaming = if r.streaming { "true" } else { "false" };
        let req = [
            r.upstream.as_str(),
            provider,
            model,
            r.outcome.as_str(),
            streaming,
        ];
        let by_model = [r.upstream.as_str(), provider, model];

        self.requests.with_label_values(&req).inc();
        self.duration
            .with_label_values(&req)
            .observe(r.timings.duration.as_secs_f64());

        let u = &r.usage;
        for (kind, n) in [
            ("input", u.input_tokens_total),
            ("output", u.output_tokens),
            ("cache_read", u.cache_read_tokens),
            ("cache_write", u.cache_write_tokens),
            ("reasoning", u.reasoning_tokens),
        ] {
            if let Some(n) = n.filter(|n| *n > 0) {
                self.tokens
                    .with_label_values(&[r.upstream.as_str(), provider, model, kind])
                    .inc_by(n);
            }
        }
        if let Some(usd) = r.cost_usd {
            self.cost.with_label_values(&by_model).inc_by(usd);
        }

        let t = &r.timings;
        for (h, v) in [
            (&self.ttfb, t.ttfb.map(|d| d.as_secs_f64())),
            (&self.ttft, t.ttft.map(|d| d.as_secs_f64())),
            (&self.output_tps, t.output_tps),
            (&self.chunk_gap, t.chunk_gap_mean.map(|d| d.as_secs_f64())),
        ] {
            if let Some(v) = v {
                h.with_label_values(&by_model).observe(v);
            }
        }

        let rl = &r.rate_limits;
        for (kind, n) in [
            ("requests", rl.requests_remaining),
            ("tokens", rl.tokens_remaining),
            ("input_tokens", rl.input_tokens_remaining),
            ("output_tokens", rl.output_tokens_remaining),
        ] {
            if let Some(n) = n {
                self.ratelimit_remaining
                    .with_label_values(&[r.upstream.as_str(), model, kind])
                    .set(n as f64);
            }
        }
        for (window, v) in [
            ("5h", rl.window_5h_utilization),
            ("7d", rl.window_7d_utilization),
        ] {
            if let Some(v) = v {
                self.window_utilization
                    .with_label_values(&[r.upstream.as_str(), window])
                    .set(v);
            }
        }
    }

    fn render(&self) -> String {
        for (module, total) in self.drops.snapshot() {
            let c = self.dropped.with_label_values(&[module]);
            c.inc_by(total.saturating_sub(c.get()));
        }
        let mut out = Vec::new();
        if let Err(e) = TextEncoder::new().encode(&self.registry.gather(), &mut out) {
            warn!(error = %e, "encoding metrics failed");
        }
        String::from_utf8(out).unwrap_or_default()
    }
}

fn register<C: Collector + Clone + 'static>(registry: &Registry, metric: C) -> Result<C> {
    registry.register(Box::new(metric.clone()))?;
    Ok(metric)
}

#[async_trait]
impl Module for Prometheus {
    fn name(&self) -> &'static str {
        "prometheus"
    }

    fn phase(&self) -> Phase {
        Phase::Export
    }

    async fn export(&self, ex: Arc<CompletedExchange>) {
        self.observe(&ex.record);
    }

    fn scrape(&self) -> Option<Scrape> {
        let this = self.clone();
        Some(Arc::new(move || this.render()))
    }
}
