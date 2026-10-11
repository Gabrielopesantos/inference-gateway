//! Eval metrics, registered on the `prometheus` module's registry. Without
//! that module (or its feature) every call is a no-op.

use anyhow::Result;

use super::judge::Decision;
use crate::modules::BuildCtx;

#[derive(Clone, Default)]
pub struct Metrics {
    #[cfg(feature = "prometheus")]
    inner: Option<Inner>,
}

#[cfg(feature = "prometheus")]
#[derive(Clone)]
struct Inner {
    jobs: ::prometheus::IntGaugeVec,
    scores: ::prometheus::IntCounterVec,
    judge_duration: ::prometheus::HistogramVec,
    confidence: ::prometheus::HistogramVec,
    failures: ::prometheus::IntCounterVec,
}

impl Metrics {
    /// # Errors
    ///
    /// Metric registration fails.
    #[cfg(feature = "prometheus")]
    pub fn new(ctx: &BuildCtx) -> Result<Self> {
        use ::prometheus::{HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts};

        let Some(registry) = &ctx.registry else {
            return Ok(Self::default());
        };
        let jobs = IntGaugeVec::new(
            Opts::new("gateway_eval_jobs", "Eval jobs by status."),
            &["status"],
        )?;
        let scores = IntCounterVec::new(
            Opts::new("gateway_eval_scores_total", "Scores stored."),
            &["evaluator", "label", "model"],
        )?;
        let judge_duration = HistogramVec::new(
            HistogramOpts::new(
                "gateway_eval_judge_duration_seconds",
                "Judge call latency, answered calls only.",
            )
            .buckets(vec![0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]),
            &["model"],
        )?;
        let confidence = HistogramVec::new(
            HistogramOpts::new(
                "gateway_eval_confidence",
                "Judge confidence of choice and score answers.",
            )
            .buckets(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            &["evaluator", "model"],
        )?;
        let failures = IntCounterVec::new(
            Opts::new(
                "gateway_eval_failures_total",
                "Failed judge attempts, retried or final.",
            ),
            &["evaluator", "reason"],
        )?;
        registry.register(Box::new(jobs.clone()))?;
        registry.register(Box::new(scores.clone()))?;
        registry.register(Box::new(judge_duration.clone()))?;
        registry.register(Box::new(confidence.clone()))?;
        registry.register(Box::new(failures.clone()))?;
        Ok(Self {
            inner: Some(Inner {
                jobs,
                scores,
                judge_duration,
                confidence,
                failures,
            }),
        })
    }

    /// # Errors
    ///
    /// Never; the signature matches the `prometheus` build.
    #[cfg(not(feature = "prometheus"))]
    #[allow(clippy::unnecessary_wraps)]
    pub fn new(_ctx: &BuildCtx) -> Result<Self> {
        Ok(Self::default())
    }

    #[allow(unused_variables)]
    pub fn jobs(&self, counts: &[(String, i64)]) {
        #[cfg(feature = "prometheus")]
        if let Some(m) = &self.inner {
            for status in ["pending", "running", "done", "failed"] {
                let n = counts
                    .iter()
                    .find(|(s, _)| s == status)
                    .map_or(0, |(_, n)| *n);
                m.jobs.with_label_values(&[status]).set(n);
            }
        }
    }

    #[allow(unused_variables)]
    pub fn scored(&self, evaluator: &str, model: &str, d: &Decision) {
        #[cfg(feature = "prometheus")]
        if let Some(m) = &self.inner {
            m.scores
                .with_label_values(&[evaluator, d.verdict.label(), model])
                .inc();
            m.judge_duration
                .with_label_values(&[d.resolved_model.as_str()])
                .observe(d.latency.as_secs_f64());
            if let Some(c) = d.confidence {
                m.confidence
                    .with_label_values(&[evaluator, model])
                    .observe(c);
            }
        }
    }

    #[allow(unused_variables)]
    pub fn failed(&self, evaluator: &str, reason: &str) {
        #[cfg(feature = "prometheus")]
        if let Some(m) = &self.inner {
            m.failures.with_label_values(&[evaluator, reason]).inc();
        }
    }
}
