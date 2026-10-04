//! The writer thread owns the only write connection. It writes whatever is
//! queued, up to [`MAX_BATCH`] exchanges, in one transaction.

use std::sync::{Arc, LazyLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use rusqlite::{Connection, Transaction, params};
use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use super::{blobs, retention};
use crate::observe::record::{CapturedContent, CompletedExchange, RequestRecord};
use crate::providers::{Operation, RequestParts};

const MAX_BATCH: usize = 100;

pub enum Msg {
    Exchange(Arc<CompletedExchange>),
    /// Delete content older than this.
    Retain(Duration),
}

/// Returns when every sender is dropped and the queue is empty.
pub fn run(mut conn: Connection, mut rx: mpsc::Receiver<Msg>) {
    let mut batch = Vec::with_capacity(MAX_BATCH);
    while let Some(first) = rx.blocking_recv() {
        let mut next = Some(first);
        while let Some(msg) = next {
            match msg {
                Msg::Exchange(ex) => batch.push(ex),
                Msg::Retain(max_age) => {
                    write_batch(&mut conn, &batch);
                    batch.clear();
                    let cutoff =
                        unix_ms(SystemTime::now().checked_sub(max_age).unwrap_or(UNIX_EPOCH));
                    if let Err(e) = retention::retain(&mut conn, cutoff) {
                        error!(error = %format!("{e:#}"), "retention pass failed");
                    }
                }
            }
            next = if batch.len() < MAX_BATCH {
                rx.try_recv().ok()
            } else {
                None
            };
        }
        write_batch(&mut conn, &batch);
        batch.clear();
    }
}

fn unix_ms(t: SystemTime) -> i64 {
    t.duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

/// A failed batch is retried one exchange per transaction, so one bad row
/// loses only itself.
fn write_batch(conn: &mut Connection, batch: &[Arc<CompletedExchange>]) {
    if batch.is_empty() {
        return;
    }
    let Err(e) = try_write(conn, batch) else {
        return;
    };
    if batch.len() == 1 {
        error!(id = %batch[0].record.id, error = %format!("{e:#}"), "trace not stored");
        return;
    }
    warn!(rows = batch.len(), error = %format!("{e:#}"), "store batch failed; retrying one by one");
    for ex in batch {
        if let Err(e) = try_write(conn, std::slice::from_ref(ex)) {
            error!(id = %ex.record.id, error = %format!("{e:#}"), "trace not stored");
        }
    }
}

fn try_write(conn: &mut Connection, batch: &[Arc<CompletedExchange>]) -> Result<()> {
    let mut tx = conn.transaction()?;
    for ex in batch {
        write_exchange(&mut tx, ex)?;
    }
    tx.commit()?;
    Ok(())
}

/// Content errors roll back only the content; the trace row stays.
fn write_exchange(tx: &mut Transaction, ex: &CompletedExchange) -> Result<()> {
    insert_trace(tx, &ex.record, ex.content.is_some())?;
    if let Some(content) = &ex.content {
        let sp = tx.savepoint()?;
        match write_content(&sp, &ex.record, content) {
            Ok(()) => sp.commit()?,
            Err(e) => {
                warn!(id = %ex.record.id, error = %format!("{e:#}"), "content not stored; trace kept");
            }
        }
    }
    Ok(())
}

const TRACE_COLUMNS: &[&str] = &[
    "id",
    "started_at",
    "upstream",
    "provider",
    "method",
    "path",
    "operation",
    "requested_model",
    "response_model",
    "streaming",
    "caller_id",
    "session_id",
    "tags",
    "http_status",
    "outcome",
    "error_type",
    "stop_reason",
    "input_tokens_total",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "cache_write_1h_tokens",
    "reasoning_tokens",
    "ttfb_ms",
    "ttft_ms",
    "ttft_text_ms",
    "duration_ms",
    "chunk_count",
    "chunk_gap_mean_ms",
    "chunk_gap_max_ms",
    "output_tps",
    "ratelimit_requests_remaining",
    "ratelimit_tokens_remaining",
    "ratelimit_input_tokens_remaining",
    "ratelimit_output_tokens_remaining",
    "retry_after_ms",
    "window_5h_utilization",
    "window_7d_utilization",
    "cost_usd",
    "captured",
];

static INSERT_TRACE: LazyLock<String> = LazyLock::new(|| {
    let placeholders: Vec<_> = (1..=TRACE_COLUMNS.len()).map(|i| format!("?{i}")).collect();
    format!(
        "INSERT INTO traces ({}) VALUES ({})",
        TRACE_COLUMNS.join(", "),
        placeholders.join(", ")
    )
});

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn insert_trace(conn: &Connection, r: &RequestRecord, captured: bool) -> Result<()> {
    let started_at = unix_ms(r.started_at);
    let (u, t, rl) = (&r.usage, &r.timings, &r.rate_limits);
    conn.prepare_cached(&INSERT_TRACE)?.execute(params![
        r.id.as_bytes().as_slice(),
        started_at,
        r.upstream,
        r.provider.as_str(),
        r.method,
        r.path,
        r.operation.map(Operation::as_str),
        r.requested_model,
        r.response_model,
        r.streaming,
        r.principal.caller_id,
        r.principal.session_id,
        serde_json::to_string(&r.principal.tags)?,
        r.http_status,
        r.outcome.as_str(),
        r.error_type,
        r.stop_reason,
        u.input_tokens_total,
        u.output_tokens,
        u.cache_read_tokens,
        u.cache_write_tokens,
        u.cache_write_1h_tokens,
        u.reasoning_tokens,
        t.ttfb.map(ms),
        t.ttft.map(ms),
        t.ttft_text.map(ms),
        ms(t.duration),
        t.chunk_count,
        t.chunk_gap_mean.map(ms),
        t.chunk_gap_max.map(ms),
        t.output_tps,
        rl.requests_remaining,
        rl.tokens_remaining,
        rl.input_tokens_remaining,
        rl.output_tokens_remaining,
        rl.retry_after.map(ms),
        rl.window_5h_utilization,
        rl.window_7d_utilization,
        r.cost_usd,
        captured,
    ])?;
    Ok(())
}

/// A body the adapter cannot split or parse still gets an output row, so
/// the raw response and truncation flag are kept.
fn write_content(conn: &Connection, r: &RequestRecord, c: &CapturedContent) -> Result<()> {
    let id = r.id.as_bytes().as_slice();
    let last_user_text = match r.provider.split_request(&c.request_body) {
        Ok(parts) => {
            write_inputs(conn, id, &parts)?;
            parts.last_user_text()
        }
        Err(e) => {
            debug!(id = %r.id, error = %e, "request body not split; inputs not stored");
            None
        }
    };
    let (blob, output_text) = if c.raw_response.is_empty() {
        (None, None)
    } else {
        let text = match r.provider.extract_output(&c.raw_response, r.streaming) {
            Ok(out) => Some(out.text),
            Err(e) => {
                debug!(id = %r.id, error = %e, "no output extracted");
                None
            }
        };
        (Some(blobs::put(conn, &c.raw_response)?), text)
    };
    conn.prepare_cached(
        "INSERT INTO trace_outputs (trace_id, blob_hash, output_text, last_user_text, truncated)
         VALUES (?1, ?2, ?3, ?4, ?5)",
    )?
    .execute(params![id, blob, output_text, last_user_text, c.truncated])?;
    Ok(())
}

fn write_inputs(conn: &Connection, id: &[u8], parts: &RequestParts) -> Result<()> {
    let mut insert = conn.prepare_cached(
        "INSERT INTO trace_inputs (trace_id, ord, kind, role, blob_hash) VALUES (?1, ?2, ?3, ?4, ?5)",
    )?;
    let fixed = [("system", &parts.system), ("tools", &parts.tools)]
        .into_iter()
        .filter_map(|(kind, v)| Some((kind, None, v.as_ref()?)));
    let messages = parts
        .messages
        .iter()
        .map(|m| ("message", m.get("role").and_then(|r| r.as_str()), m));
    for (ord, (kind, role, value)) in fixed.chain(messages).enumerate() {
        insert.execute(params![id, ord, kind, role, blobs::put_json(conn, value)?])?;
    }
    conn.prepare_cached("INSERT INTO trace_params (trace_id, json) VALUES (?1, ?2)")?
        .execute(params![id, serde_json::to_string(&parts.params)?])?;
    Ok(())
}
