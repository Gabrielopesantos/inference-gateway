//! Claims jobs, renders state, asks the judge and stores the score. One
//! claim loop; at most `concurrency` jobs in flight.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use rusqlite::Connection;
use tokio::sync::{Notify, watch};
use tokio::task::{self, JoinSet};
use tokio::time::Instant;
use tracing::{debug, error, warn};
use uuid::Uuid;

use super::jev::{Decision, Judge};
use super::metrics::Metrics;
use super::queue::{self, Job, Score};
use super::template::{self, Vars};
use crate::config::eval::{EvalConfig, Evaluator};
use crate::store::{Store, unix_ms};

/// How often the job-count gauge is refreshed.
const COUNTS_EVERY: Duration = Duration::from_secs(10);

pub struct Worker {
    /// Queue writes; claims and updates are short.
    conn: Arc<Mutex<Connection>>,
    store: Store,
    judge: Judge,
    /// Configured evaluators by name; jobs of other versions are not claimed.
    evaluators: HashMap<String, Evaluator>,
    versions: Vec<String>,
    concurrency: usize,
    max_attempts: u32,
    lease: Duration,
    poll: Duration,
    max_state_tokens: usize,
    metrics: Metrics,
}

fn now_ms() -> i64 {
    unix_ms(SystemTime::now())
}

impl Worker {
    /// # Errors
    ///
    /// Database cannot be opened or written.
    pub fn new(cfg: &EvalConfig, store: Store, judge: Judge, metrics: Metrics) -> Result<Self> {
        let mut conn = crate::store::open_writer(store.path())?;
        queue::register(&mut conn, &cfg.evaluators, now_ms()).context("registering evaluators")?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            store,
            judge,
            evaluators: cfg
                .evaluators
                .iter()
                .map(|e| (e.name.clone(), e.clone()))
                .collect(),
            versions: cfg.evaluators.iter().map(|e| e.version.clone()).collect(),
            concurrency: cfg.concurrency,
            max_attempts: cfg.max_attempts,
            lease: cfg.lease,
            poll: cfg.poll,
            max_state_tokens: cfg.judge.max_state_tokens,
            metrics,
        })
    }

    /// Runs a queue call on the blocking pool.
    async fn db<R, F>(&self, f: F) -> Result<R>
    where
        R: Send + 'static,
        F: FnOnce(&mut Connection) -> Result<R> + Send + 'static,
    {
        let conn = self.conn.clone();
        task::spawn_blocking(move || f(&mut conn.lock().unwrap_or_else(PoisonError::into_inner)))
            .await
            .context("queue task")?
    }

    /// Until `stop` turns true. Then in-flight jobs are cancelled and
    /// released, so another worker or the next start picks them up.
    pub async fn run(self: Arc<Self>, notify: Arc<Notify>, mut stop: watch::Receiver<bool>) {
        let mut tasks = JoinSet::new();
        let mut in_flight: HashMap<task::Id, Job> = HashMap::new();
        let mut counted: Option<Instant> = None;
        loop {
            if *stop.borrow() {
                break;
            }
            let free = self.concurrency - tasks.len();
            if free > 0 {
                let (versions, lease) = (self.versions.clone(), self.lease);
                match self
                    .db(move |c| queue::claim(c, &versions, free, lease, now_ms()))
                    .await
                {
                    Ok(jobs) => {
                        for job in jobs {
                            let this = self.clone();
                            let j = job.clone();
                            let handle = tasks.spawn(async move { this.process(j).await });
                            in_flight.insert(handle.id(), job);
                        }
                    }
                    Err(e) => error!(error = %format!("{e:#}"), "claiming eval jobs failed"),
                }
            }
            if counted.is_none_or(|t| t.elapsed() >= COUNTS_EVERY) {
                counted = Some(Instant::now());
                match self.db(|c| queue::counts(c)).await {
                    Ok(counts) => self.metrics.jobs(&counts),
                    Err(e) => warn!(error = %format!("{e:#}"), "counting eval jobs failed"),
                }
            }
            tokio::select! {
                _ = stop.changed() => break,
                () = notify.notified() => {}
                () = tokio::time::sleep(self.poll) => {}
                Some(done) = tasks.join_next_with_id() => {
                    let id = match done {
                        Ok((id, ())) => id,
                        Err(e) => {
                            error!(error = %e, "eval job task failed");
                            e.id()
                        }
                    };
                    in_flight.remove(&id);
                }
            }
        }

        tasks.abort_all();
        while let Some(done) = tasks.join_next_with_id().await {
            let (id, cancelled) = match done {
                Ok((id, ())) => (id, false),
                Err(e) => (e.id(), e.is_cancelled()),
            };
            if let Some(job) = in_flight.remove(&id)
                && cancelled
            {
                debug!(job = job.id, "releasing eval job on shutdown");
                if let Err(e) = self.db(move |c| queue::release(c, &job, now_ms())).await {
                    warn!(error = %format!("{e:#}"), "releasing eval job failed; its lease will expire");
                }
            }
        }
    }

    async fn process(self: Arc<Self>, job: Job) {
        let outcome = self.attempt(&job).await;
        let result = match outcome {
            Attempt::Scored(score, decision, model) => {
                let j = job.clone();
                match self
                    .db(move |c| queue::complete(c, &j, &score, now_ms()))
                    .await
                {
                    Ok(true) => {
                        self.metrics.scored(&job.evaluator, &model, &decision);
                        Ok(())
                    }
                    Ok(false) => {
                        debug!(job = job.id, "lease lost; score dropped");
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }
            Attempt::Retry {
                reason,
                error,
                after,
            } => {
                self.metrics.failed(&job.evaluator, &reason);
                let delay = queue::backoff(&job, after);
                let (j, max) = (job.clone(), self.max_attempts);
                self.db(move |c| queue::retry(c, &j, max, delay, &error, now_ms()))
                    .await
                    .map(|failed| {
                        if failed {
                            warn!(
                                job = job.id,
                                evaluator = job.evaluator,
                                reason,
                                "eval job failed after max attempts"
                            );
                        }
                    })
            }
            Attempt::Fail { reason, error } => {
                self.metrics.failed(&job.evaluator, &reason);
                warn!(
                    job = job.id,
                    evaluator = job.evaluator,
                    reason,
                    error,
                    "eval job failed"
                );
                let j = job.clone();
                self.db(move |c| queue::fail(c, &j, &error)).await
            }
        };
        if let Err(e) = result {
            error!(job = job.id, error = %format!("{e:#}"), "updating eval job failed; its lease will expire");
        }
    }

    async fn attempt(&self, job: &Job) -> Attempt {
        let Some(e) = self
            .evaluators
            .get(&job.evaluator)
            .filter(|e| e.version == job.evaluator_version)
        else {
            return Attempt::fail("config", "evaluator version not configured");
        };
        let budget = template::budget_chars(self.max_state_tokens, &e.name, &e.question);
        let (store, trace_id, template, id) = (
            self.store.clone(),
            job.trace_id.clone(),
            e.state.clone(),
            job.id,
        );
        // Content can be megabytes: load and render off the async runtime,
        // and outside the queue lock.
        let prepared =
            task::spawn_blocking(move || prepare(&store, &trace_id, &template, budget, id))
                .await
                .context("render task")
                .and_then(|r| r);
        let (state, truncated, model) = match prepared {
            Ok(Prepared::State {
                state,
                truncated,
                model,
            }) => (state, truncated, model),
            Ok(Prepared::Missing) => {
                return Attempt::fail("expired", "trace content not captured or expired");
            }
            Ok(Prepared::BadTemplate(err)) => return Attempt::fail("render", &err),
            Err(e) => {
                return Attempt::Retry {
                    reason: "store".into(),
                    error: format!("{e:#}"),
                    after: None,
                };
            }
        };
        match self.judge.ask(&e.name, &state, &e.question).await {
            Ok(d) => Attempt::Scored(
                score_row(&d, truncated),
                d,
                model.unwrap_or_else(|| "unknown".into()),
            ),
            Err(err) if err.retryable => Attempt::Retry {
                reason: err.reason,
                error: err.message,
                after: err.retry_after,
            },
            Err(err) => Attempt::Fail {
                reason: err.reason,
                error: err.message,
            },
        }
    }
}

enum Prepared {
    /// With the trace's model, for metrics.
    State {
        state: String,
        truncated: bool,
        model: Option<String>,
    },
    /// Content not captured, expired, or the trace row is gone.
    Missing,
    BadTemplate(String),
}

/// Loads the trace and renders the state. Blocking.
fn prepare(
    store: &Store,
    trace_id: &[u8],
    template: &str,
    budget: usize,
    job_id: i64,
) -> Result<Prepared> {
    let meta = queue::trace_meta(&store.reader()?, trace_id)?;
    let content = store.load_trace_content(Uuid::from_slice(trace_id)?)?;
    let (Some(content), Some(meta)) = (content, meta) else {
        return Ok(Prepared::Missing);
    };
    let output = (!content.raw_response.is_empty())
        .then(|| {
            meta.provider
                .extract_output(&content.raw_response, meta.streaming)
                .inspect_err(|e| debug!(job = job_id, error = %e, "no output extracted"))
                .ok()
        })
        .flatten();
    let vars = Vars::from_content(&content, output.as_ref(), meta.stop_reason.as_deref());
    Ok(match template::render(template, &vars, budget) {
        Ok((state, truncated)) => Prepared::State {
            state,
            truncated,
            model: meta.model,
        },
        Err(err) => Prepared::BadTemplate(format!("state template: {err:#}")),
    })
}

// Short-lived; boxing would only add an allocation per job.
#[allow(clippy::large_enum_variant)]
enum Attempt {
    /// With the trace's model, for metrics.
    Scored(Score, Decision, String),
    Retry {
        reason: String,
        error: String,
        after: Option<Duration>,
    },
    Fail {
        reason: String,
        error: String,
    },
}

impl Attempt {
    fn fail(reason: &str, error: &str) -> Self {
        Self::Fail {
            reason: reason.into(),
            error: error.into(),
        }
    }
}

fn score_row(d: &Decision, truncated: bool) -> Score {
    let v = &d.verdict;
    Score {
        data_type: v.data_type(),
        value_bool: v.value_bool(),
        value_label: v.value_label().map(str::to_owned),
        value_num: v.value_num(),
        confidence: d.confidence,
        distribution: serde_json::to_string(&d.distribution).unwrap_or_default(),
        judge_model: d.resolved_model.clone(),
        judge_latency_ms: d.latency.as_secs_f64() * 1000.0,
        judge_input_tokens: d.input_tokens,
        judge_cost_usd: d.cost_usd,
        input_truncated: truncated,
    }
}
