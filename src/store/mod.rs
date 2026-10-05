//! `SQLite` store: one row per trace, plus captured content split into
//! deduplicated blobs. The `store` module feeds a writer thread that owns the
//! only write connection; readers open their own connections.

mod blobs;
mod reader;
mod retention;
mod schema;
mod writer;

pub use self::reader::TraceContent;
pub use self::schema::{open_reader, open_writer};
pub use self::writer::{Jobs, enqueue, unix_ms};

use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use rusqlite::Connection;
use tokio::sync::mpsc;
use tracing::error;

use self::writer::Msg;
use crate::config::StoreFile;
use crate::modules::{CompletedExchange, Module, Phase};

/// Between the export worker and the writer thread. When full, the export
/// worker waits and the dispatcher queue absorbs, then counts, the overflow.
const CHANNEL_CAPACITY: usize = 256;

/// Read access for other modules.
#[derive(Debug, Clone)]
pub struct Store {
    path: PathBuf,
}

impl Store {
    /// For processes without the `store` module: creates and migrates the
    /// database if needed.
    ///
    /// # Errors
    ///
    /// Database cannot be opened or migrated.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        drop(schema::open_writer(path)?);
        Ok(Self {
            path: path.to_owned(),
        })
    }

    /// # Errors
    ///
    /// Database missing or unreadable.
    pub fn reader(&self) -> Result<Connection> {
        schema::open_reader(&self.path)
    }

    #[must_use]
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }
}

pub struct StoreModule {
    /// Taken on shutdown, which ends the writer thread.
    tx: Mutex<Option<mpsc::Sender<Msg>>>,
    writer: Mutex<Option<JoinHandle<()>>>,
    /// Holds a sender too, so it is stopped first.
    retention: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

impl StoreModule {
    /// Opens the database, creating and migrating it, and starts the writer
    /// and the retention timer. Call inside a tokio runtime.
    ///
    /// # Errors
    ///
    /// Database cannot be opened or migrated, or the thread fails to spawn.
    pub fn start(cfg: &StoreFile, retention: Duration, jobs: Jobs) -> Result<(Self, Store)> {
        let conn = schema::open_writer(&cfg.path)?;
        let (tx, rx) = mpsc::channel(CHANNEL_CAPACITY);
        let writer = std::thread::Builder::new()
            .name("store-writer".into())
            .spawn(move || writer::run(conn, rx, &jobs))
            .context("spawning store writer")?;
        let timer = tokio::spawn(retention::run(tx.clone(), retention));
        let module = Self {
            tx: Mutex::new(Some(tx)),
            writer: Mutex::new(Some(writer)),
            retention: Mutex::new(Some(timer)),
        };
        Ok((
            module,
            Store {
                path: cfg.path.clone(),
            },
        ))
    }

    fn sender(&self) -> Option<mpsc::Sender<Msg>> {
        self.tx
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

#[async_trait]
impl Module for StoreModule {
    fn name(&self) -> &'static str {
        "store"
    }

    fn phase(&self) -> Phase {
        Phase::Export
    }

    async fn export(&self, ex: Arc<CompletedExchange>) {
        let Some(tx) = self.sender() else { return };
        if tx.send(Msg::Exchange(ex)).await.is_err() {
            error!("store writer stopped; trace dropped");
        }
    }

    /// Waits for the writer to commit everything queued.
    async fn shutdown(&self) {
        let timer = self
            .retention
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(timer) = timer {
            timer.abort();
            let _ = timer.await;
        }
        drop(
            self.tx
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take(),
        );
        let writer = self
            .writer
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if let Some(writer) = writer
            && !matches!(
                tokio::task::spawn_blocking(move || writer.join()).await,
                Ok(Ok(()))
            )
        {
            error!("store writer panicked");
        }
    }
}
