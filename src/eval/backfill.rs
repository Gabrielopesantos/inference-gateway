//! Enqueues jobs for traces captured before an evaluator (or its current
//! version) existed.

use std::collections::BTreeMap;

use anyhow::Result;
use rusqlite::Connection;

use crate::capture::TraceFacts;
use crate::config::eval::Evaluator;
use crate::providers;
use crate::store::enqueue;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Report {
    /// Captured `ok` traces in range that match the evaluator.
    pub matched: usize,
    /// Of those, picked by the evaluator's `sample`.
    pub sampled: usize,
    /// Of those, without a job for the current version yet.
    pub enqueued: usize,
}

/// Same selection as live enqueue: content stored, outcome `ok`, `match`
/// and (unless `all`) `sample`. A dry run rolls back.
///
/// # Errors
///
/// Database error.
pub fn backfill(
    conn: &mut Connection,
    e: &Evaluator,
    since_ms: i64,
    all: bool,
    dry_run: bool,
    now_ms: i64,
) -> Result<Report> {
    let tx = conn.transaction()?;
    let mut report = Report::default();
    {
        let mut stmt = tx.prepare(
            "SELECT t.id, t.upstream, coalesce(t.response_model, t.requested_model), t.tags,
                    t.stop_reason, p.json
               FROM traces t JOIN trace_outputs o ON o.trace_id = t.id
                    LEFT JOIN trace_params p ON p.trace_id = t.id
              WHERE t.outcome = 'ok' AND t.started_at >= ?1
              ORDER BY t.started_at",
        )?;
        let mut rows = stmt.query([since_ms])?;
        while let Some(r) = rows.next()? {
            let id: Vec<u8> = r.get(0)?;
            let upstream: String = r.get(1)?;
            let model: Option<String> = r.get(2)?;
            let tags: BTreeMap<String, String> = serde_json::from_str(&r.get::<_, String>(3)?)?;
            let stop_reason: Option<String> = r.get(4)?;
            // No params row: the body could not be split, as on live enqueue.
            let params: Option<String> = r.get(5)?;
            let structured_output = params
                .map(|p| serde_json::from_str(&p))
                .transpose()?
                .is_some_and(|p| providers::structured_output(&p));
            let facts = TraceFacts {
                stop_reason: stop_reason.as_deref(),
                structured_output,
            };
            if !e
                .matcher
                .matches(&upstream, model.as_deref(), &tags, Some(facts))
            {
                continue;
            }
            report.matched += 1;
            if !all && !e.sampled(&id) {
                continue;
            }
            report.sampled += 1;
            report.enqueued += enqueue(&tx, &id, e, now_ms)?;
        }
    }
    if !dry_run {
        tx.commit()?;
    }
    Ok(report)
}
