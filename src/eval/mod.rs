//! Asynchronous scoring of captured traces by the Jev judge. The store
//! writer enqueues jobs with the trace; a worker drains the queue, here or
//! in `inference-gateway eval run` against the same database.

pub mod backfill;
pub mod jev;
pub mod metrics;
pub mod queue;
pub mod template;
pub mod worker;

use std::sync::{Arc, Mutex, PoisonError};

use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio::sync::{Notify, watch};
use tokio::task::JoinHandle;
use tracing::error;

use self::jev::Judge;
use self::metrics::Metrics;
use self::worker::Worker;
use crate::config::eval::{EvalConfig, Judge as JudgeConfig};
use crate::modules::{BuildCtx, Module, Phase};
use crate::store::Store;

pub struct EvalModule {
    stop: watch::Sender<bool>,
    task: Mutex<Option<JoinHandle<()>>>,
}

/// # Errors
///
/// The env var named by `api_key_env` is unset or empty.
pub fn judge_key(cfg: &JudgeConfig) -> Result<String> {
    std::env::var(&cfg.api_key_env)
        .ok()
        .filter(|k| !k.is_empty())
        .with_context(|| format!("eval.judge.api_key_env: {} is not set", cfg.api_key_env))
}

impl EvalModule {
    /// From the shared handles of the `store` and `prometheus` modules.
    ///
    /// # Errors
    ///
    /// No store, no judge key, or the worker fails to start.
    pub fn start(cfg: &EvalConfig, ctx: &BuildCtx) -> Result<Self> {
        let store = ctx
            .store
            .clone()
            .context("[eval] needs a [store] section")?;
        let key = match &ctx.judge_key {
            Some(k) => k.clone(),
            None => judge_key(&cfg.judge)?,
        };
        let notify = ctx.jobs_notify.clone().unwrap_or_default();
        Self::spawn(cfg, store, key, notify, Metrics::new(ctx)?)
    }

    /// Starts the worker; call inside a tokio runtime.
    ///
    /// # Errors
    ///
    /// Judge client or queue connection fails to build.
    pub fn spawn(
        cfg: &EvalConfig,
        store: Store,
        key: String,
        notify: Arc<Notify>,
        metrics: Metrics,
    ) -> Result<Self> {
        let judge = Judge::new(&cfg.judge, key)?;
        let worker = Arc::new(Worker::new(cfg, store, judge, metrics)?);
        let (stop, stopped) = watch::channel(false);
        let task = tokio::spawn(worker.run(notify, stopped));
        Ok(Self {
            stop,
            task: Mutex::new(Some(task)),
        })
    }
}

#[async_trait]
impl Module for EvalModule {
    fn name(&self) -> &'static str {
        "eval"
    }

    fn phase(&self) -> Phase {
        Phase::Background
    }

    /// Stops claiming and releases in-flight jobs.
    async fn shutdown(&self) {
        let _ = self.stop.send(true);
        let task = self
            .task
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(task) = task
            && task.await.is_err()
        {
            error!("eval worker panicked");
        }
    }
}
