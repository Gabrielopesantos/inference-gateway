//! Consumers of finished records. Enrich modules run in order on the proxy
//! path; each export module has its own worker behind a bounded queue, so a
//! slow export drops records instead of slowing the proxy.

pub mod access_log;
#[cfg(feature = "cost")]
pub mod cost;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Enrich,
    Export,
}

#[derive(Debug)]
pub struct CompletedExchange {
    pub record: RequestRecord,
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
pub fn build(cfg: &Config, ctx: &BuildCtx) -> Result<Vec<Box<dyn Module>>> {
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
    #[cfg(not(feature = "prometheus"))]
    let _ = ctx;
    Ok(modules)
}

pub struct Dispatcher {
    inner: Arc<Inner>,
    workers: Vec<JoinHandle<()>>,
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
        for module in modules {
            match module.phase() {
                Phase::Enrich => enrich.push(module),
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
            scrape,
        }
    }

    #[must_use]
    pub fn sink(&self) -> RecordSink {
        let inner = self.inner.clone();
        Arc::new(move |record| inner.dispatch(record))
    }

    #[must_use]
    pub fn scrape(&self) -> Option<Scrape> {
        self.scrape.clone()
    }

    /// Closes the queues, then waits up to `deadline` for workers to drain.
    pub async fn shutdown(self, deadline: Duration) {
        drop(std::mem::take(
            &mut *self
                .inner
                .exports
                .write()
                .unwrap_or_else(PoisonError::into_inner),
        ));
        if tokio::time::timeout(deadline, join_all(self.workers))
            .await
            .is_err()
        {
            warn!(?deadline, "export modules did not drain in time");
        }
    }
}

impl Inner {
    fn dispatch(&self, mut record: RequestRecord) {
        for m in &self.enrich {
            m.enrich(&mut record);
        }
        let ex = Arc::new(CompletedExchange { record });
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
        let modules = build(&Config::parse(MINIMAL).unwrap(), &BuildCtx::default()).unwrap();
        let names: Vec<_> = modules.iter().map(|m| m.name()).collect();
        assert_eq!(names, ["access_log"]);
    }

    #[cfg(not(feature = "prometheus"))]
    #[test]
    fn prometheus_section_without_feature_fails() {
        let cfg = Config::parse(&format!("{MINIMAL}\n[prometheus]\n")).unwrap();
        let err = build(&cfg, &BuildCtx::default()).err().unwrap();
        assert_eq!(
            err.to_string(),
            "[prometheus] needs cargo feature `prometheus`"
        );
    }

    #[cfg(not(feature = "cost"))]
    #[test]
    fn cost_section_without_feature_fails() {
        let cfg =
            Config::parse(&format!("{MINIMAL}\n[cost]\npricing_file = \"p.toml\"\n")).unwrap();
        let err = build(&cfg, &BuildCtx::default()).err().unwrap();
        assert_eq!(err.to_string(), "[cost] needs cargo feature `cost`");
    }
}
