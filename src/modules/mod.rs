//! Consumers of finished records. Enrich modules run in order on the proxy
//! path; each export module has its own worker behind a bounded queue, so a
//! slow export drops records instead of slowing the proxy.

pub mod access_log;
#[cfg(feature = "cost")]
pub mod cost;
#[cfg(feature = "otel")]
pub mod otel;
#[cfg(feature = "prometheus")]
pub mod prometheus;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use futures_util::future::join_all;
use tokio::sync::mpsc::{self, error::TrySendError};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use crate::config::Config;
use crate::observe::RecordSink;
use crate::observe::record::RequestRecord;
pub use crate::observe::record::{CapturedContent, CompletedExchange};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Enrich,
    Export,
    /// Runs its own tasks and sees no records; only `shutdown` is called.
    Background,
}

/// Renders the admin `/metrics` body.
pub type Scrape = Arc<dyn Fn() -> String + Send + Sync>;

#[async_trait]
pub trait Module: Send + Sync {
    fn name(&self) -> &'static str;
    fn phase(&self) -> Phase;
    /// Must be cheap: runs on the proxy path.
    fn enrich(&self, _rec: &mut RequestRecord) {}
    async fn export(&self, _ex: Arc<CompletedExchange>) {}
    /// Flush; runs after the queue drains.
    async fn shutdown(&self) {}
    fn scrape(&self) -> Option<Scrape> {
        None
    }
}

/// Shared handles for module constructors.
#[derive(Default)]
pub struct BuildCtx {
    pub drops: Arc<DropCounters>,
    /// Set by the `store` module.
    #[cfg(feature = "store")]
    pub store: Option<crate::store::Store>,
    /// Set by the `store` module when `[eval]` is present; woken when the
    /// writer enqueues eval jobs.
    #[cfg(feature = "store")]
    pub jobs_notify: Option<Arc<tokio::sync::Notify>>,
    /// Set by the `prometheus` module, for other modules' metrics.
    #[cfg(feature = "prometheus")]
    pub registry: Option<::prometheus::Registry>,
    /// Judge key; read from the env var named in `[eval.judge]` when unset.
    #[cfg(feature = "eval")]
    pub judge_key: Option<String>,
}

/// Records dropped per export module because its queue was full.
#[derive(Default)]
pub struct DropCounters(Mutex<Vec<(&'static str, Arc<AtomicU64>)>>);

impl DropCounters {
    fn register(&self, module: &'static str) -> Arc<AtomicU64> {
        let counter = Arc::new(AtomicU64::new(0));
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((module, counter.clone()));
        counter
    }

    #[must_use]
    pub fn snapshot(&self) -> Vec<(&'static str, u64)> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .map(|(name, c)| (*name, c.load(Ordering::Relaxed)))
            .collect()
    }
}

/// One arm per config section; a present section enables the module.
///
/// # Errors
///
/// A module fails to build, or its section is present but its feature is off.
pub fn build(cfg: &Config, ctx: &mut BuildCtx) -> Result<Vec<Box<dyn Module>>> {
    // Before the store starts its writer.
    #[cfg(not(feature = "eval"))]
    if cfg.eval.is_some() {
        anyhow::bail!("[eval] needs cargo feature `eval`");
    }
    #[allow(unused_mut)] // with every feature off, nothing is pushed
    let mut modules: Vec<Box<dyn Module>> = vec![Box::new(access_log::AccessLog)];
    if let Some(c) = &cfg.cost {
        #[cfg(feature = "cost")]
        modules.push(Box::new(cost::Cost::new(c)?));
        #[cfg(not(feature = "cost"))]
        {
            let _ = c;
            anyhow::bail!("[cost] needs cargo feature `cost`");
        }
    }
    if let Some(c) = &cfg.prometheus {
        #[cfg(feature = "prometheus")]
        modules.push(Box::new(prometheus::Prometheus::new(c, ctx)?));
        #[cfg(not(feature = "prometheus"))]
        {
            let _ = c;
            anyhow::bail!("[prometheus] needs cargo feature `prometheus`");
        }
    }
    if let Some(c) = &cfg.store {
        #[cfg(feature = "store")]
        {
            let retention = cfg
                .capture
                .as_ref()
                .map_or(crate::config::DEFAULT_RETENTION, |p| p.retention);
            let mut jobs = crate::store::Jobs::default();
            if let Some(e) = &cfg.eval {
                let notify = Arc::new(tokio::sync::Notify::new());
                jobs.evaluators = e.evaluators.clone().into();
                jobs.notify = Some(notify.clone());
                ctx.jobs_notify = Some(notify);
            }
            let (module, store) = crate::store::StoreModule::start(c, retention, jobs)?;
            ctx.store = Some(store);
            modules.push(Box::new(module));
        }
        #[cfg(not(feature = "store"))]
        {
            let _ = c;
            anyhow::bail!("[store] needs cargo feature `store`");
        }
    }
    #[cfg(feature = "eval")]
    if let Some(c) = &cfg.eval {
        modules.push(Box::new(crate::eval::EvalModule::start(c, ctx)?));
    }
    if let Some(c) = &cfg.otel {
        #[cfg(feature = "otel")]
        modules.push(Box::new(otel::Otel::new(c)?));
        #[cfg(not(feature = "otel"))]
        {
            let _ = c;
            anyhow::bail!("[otel] needs cargo feature `otel`");
        }
    }
    #[cfg(not(any(feature = "prometheus", feature = "store")))]
    let _ = ctx;
    Ok(modules)
}

pub struct Dispatcher {
    inner: Arc<Inner>,
    workers: Vec<JoinHandle<()>>,
    background: Vec<Box<dyn Module>>,
    scrape: Option<Scrape>,
}

struct Inner {
    enrich: Vec<Box<dyn Module>>,
    /// Emptied on shutdown, which closes the queues.
    exports: RwLock<Vec<Queue>>,
}

struct Queue {
    name: &'static str,
    tx: mpsc::Sender<Arc<CompletedExchange>>,
    dropped: Arc<AtomicU64>,
}

impl Dispatcher {
    pub const QUEUE_CAPACITY: usize = 1024;

    /// Spawns one worker per export module; call inside a tokio runtime.
    #[must_use]
    pub fn start(modules: Vec<Box<dyn Module>>, ctx: &BuildCtx, capacity: usize) -> Self {
        let scrape = modules.iter().find_map(|m| m.scrape());
        let mut enrich = Vec::new();
        let mut exports = Vec::new();
        let mut workers = Vec::new();
        let mut background = Vec::new();
        for module in modules {
            match module.phase() {
                Phase::Enrich => enrich.push(module),
                Phase::Background => background.push(module),
                Phase::Export => {
                    let name = module.name();
                    let (tx, mut rx) = mpsc::channel::<Arc<CompletedExchange>>(capacity);
                    workers.push(tokio::spawn(async move {
                        while let Some(ex) = rx.recv().await {
                            module.export(ex).await;
                        }
                        module.shutdown().await;
                    }));
                    exports.push(Queue {
                        name,
                        tx,
                        dropped: ctx.drops.register(name),
                    });
                }
            }
        }
        Self {
            inner: Arc::new(Inner {
                enrich,
                exports: RwLock::new(exports),
            }),
            workers,
            background,
            scrape,
        }
    }

    #[must_use]
    pub fn sink(&self) -> RecordSink {
        let inner = self.inner.clone();
        Arc::new(move |ex| inner.dispatch(ex))
    }

    #[must_use]
    pub fn scrape(&self) -> Option<Scrape> {
        self.scrape.clone()
    }

    /// Closes the queues, then waits up to `deadline` for workers to drain
    /// and background modules to stop.
    pub async fn shutdown(self, deadline: Duration) {
        drop(std::mem::take(
            &mut *self
                .inner
                .exports
                .write()
                .unwrap_or_else(PoisonError::into_inner),
        ));
        let background = join_all(self.background.iter().map(|m| m.shutdown()));
        if tokio::time::timeout(deadline, async {
            tokio::join!(join_all(self.workers), background)
        })
        .await
        .is_err()
        {
            warn!(?deadline, "modules did not shut down in time");
        }
    }
}

impl Inner {
    fn dispatch(&self, mut ex: CompletedExchange) {
        for m in &self.enrich {
            m.enrich(&mut ex.record);
        }
        let ex = Arc::new(ex);
        let exports = self.exports.read().unwrap_or_else(PoisonError::into_inner);
        for q in exports.iter() {
            if let Err(TrySendError::Full(_)) = q.tx.try_send(ex.clone()) {
                q.dropped.fetch_add(1, Ordering::Relaxed);
                debug!(module = q.name, id = %ex.record.id, "export queue full; record dropped");
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::config::Config;

    const MINIMAL: &str =
        "[[upstreams]]\nname = \"a\"\nprovider = \"openai\"\nbase_url = \"https://x\"\n";

    #[test]
    fn access_log_is_always_on() {
        let modules = build(&Config::parse(MINIMAL).unwrap(), &mut BuildCtx::default()).unwrap();
        let names: Vec<_> = modules.iter().map(|m| m.name()).collect();
        assert_eq!(names, ["access_log"]);
    }

    #[cfg(not(feature = "prometheus"))]
    #[test]
    fn prometheus_section_without_feature_fails() {
        let cfg = Config::parse(&format!("{MINIMAL}\n[prometheus]\n")).unwrap();
        let err = build(&cfg, &mut BuildCtx::default()).err().unwrap();
        assert_eq!(
            err.to_string(),
            "[prometheus] needs cargo feature `prometheus`"
        );
    }

    #[cfg(not(feature = "cost"))]
    #[test]
    fn cost_section_without_feature_fails() {
        let cfg = Config::parse(&format!("{MINIMAL}\n[cost]\n")).unwrap();
        let err = build(&cfg, &mut BuildCtx::default()).err().unwrap();
        assert_eq!(err.to_string(), "[cost] needs cargo feature `cost`");
    }

    #[cfg(not(feature = "store"))]
    #[test]
    fn store_section_without_feature_fails() {
        let cfg = Config::parse(&format!("{MINIMAL}\n[store]\npath = \"g.db\"\n")).unwrap();
        let err = build(&cfg, &mut BuildCtx::default()).err().unwrap();
        assert_eq!(err.to_string(), "[store] needs cargo feature `store`");
    }

    #[cfg(not(feature = "otel"))]
    #[test]
    fn otel_section_without_feature_fails() {
        let cfg = Config::parse(&format!("{MINIMAL}\n[otel]\nendpoint = \"http://c\"\n")).unwrap();
        let err = build(&cfg, &mut BuildCtx::default()).err().unwrap();
        assert_eq!(err.to_string(), "[otel] needs cargo feature `otel`");
    }

    #[cfg(not(feature = "eval"))]
    #[test]
    fn eval_section_without_feature_fails() {
        let cfg = Config::parse(&format!(
            "{MINIMAL}\n[store]\npath = \"g.db\"\n\n[eval.judge]\nbase_url = \"https://j\"\nmodel = \"m\"\napi_key_env = \"K\"\n"
        ))
        .unwrap();
        let err = build(&cfg, &mut BuildCtx::default()).err().unwrap();
        assert_eq!(err.to_string(), "[eval] needs cargo feature `eval`");
    }
}
