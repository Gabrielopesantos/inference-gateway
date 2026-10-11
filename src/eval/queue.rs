//! Durable job queue in the `eval_jobs` table. A claim leases a job; an
//! expired lease lets another worker take it, so a crash only causes a retry.
//! Every write after the claim is guarded by `(status, attempts)`, so a
//! worker whose lease was taken over cannot finish the job a second time.
//!
//! Functions take `now_ms` (unix ms), so tests control the clock.

use std::time::Duration;

use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use sha2::{Digest, Sha256};

use crate::config::eval::Evaluator;
use crate::providers::ProviderKind;

const BACKOFF_BASE: Duration = Duration::from_secs(1);
const BACKOFF_MAX: Duration = Duration::from_mins(5);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Job {
    pub id: i64,
    pub trace_id: Vec<u8>,
    pub evaluator: String,
    pub evaluator_version: String,
    /// Including this claim.
    pub attempts: u32,
}

/// What a finished job stores; mirrors the `scores` columns.
#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    pub data_type: &'static str,
    pub value_bool: Option<bool>,
    pub value_label: Option<String>,
    pub value_num: Option<f64>,
    pub confidence: Option<f64>,
    /// JSON object.
    pub distribution: String,
    pub judge_model: String,
    pub judge_latency_ms: f64,
    pub judge_input_tokens: u64,
    pub judge_cost_usd: f64,
    pub input_truncated: bool,
}

fn ms(d: Duration) -> i64 {
    i64::try_from(d.as_millis()).unwrap_or(i64::MAX)
}

/// Leases up to `n` due jobs of the given evaluator versions: pending ones
/// past `not_before`, and running ones whose lease expired. Jobs of versions
/// not configured here stay queued. An expired lease on the last attempt
/// fails the job, so a job that kills its task cannot loop forever.
///
/// # Errors
///
/// Database error.
pub fn claim(
    conn: &Connection,
    versions: &[String],
    n: usize,
    lease: Duration,
    max_attempts: u32,
    now_ms: i64,
) -> Result<Vec<Job>> {
    conn.prepare_cached(
        "UPDATE eval_jobs
            SET status = 'failed', lease_until = NULL,
                last_error = 'lease expired on the last attempt'
          WHERE status = 'running' AND lease_until < ?1 AND attempts >= ?2",
    )?
    .execute(params![now_ms, max_attempts])?;
    let mut stmt = conn.prepare_cached(
        "UPDATE eval_jobs
            SET status = 'running', attempts = attempts + 1, lease_until = ?1 + ?2
          WHERE id IN (SELECT id FROM eval_jobs
                        WHERE ((status = 'pending' AND not_before <= ?1)
                            OR (status = 'running' AND lease_until < ?1))
                          AND evaluator_version IN (SELECT value FROM json_each(?3))
                        ORDER BY id LIMIT ?4)
        RETURNING id, trace_id, evaluator, evaluator_version, attempts",
    )?;
    let jobs = stmt
        .query_map(
            params![
                now_ms,
                ms(lease),
                serde_json::to_string(versions)?,
                i64::try_from(n).unwrap_or(i64::MAX)
            ],
            |r| {
                Ok(Job {
                    id: r.get(0)?,
                    trace_id: r.get(1)?,
                    evaluator: r.get(2)?,
                    evaluator_version: r.get(3)?,
                    attempts: r.get(4)?,
                })
            },
        )?
        .collect::<Result<_, _>>()?;
    Ok(jobs)
}

/// What rendering and metrics need from the trace row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceMeta {
    /// Response model, else requested.
    pub model: Option<String>,
    pub provider: ProviderKind,
    pub streaming: bool,
    pub stop_reason: Option<String>,
}

/// `None` when the trace row is gone.
///
/// # Errors
///
/// Database error, or a provider this build does not know.
pub fn trace_meta(conn: &Connection, trace_id: &[u8]) -> Result<Option<TraceMeta>> {
    let row = conn
        .query_row(
            "SELECT coalesce(response_model, requested_model), provider, streaming, stop_reason
               FROM traces WHERE id = ?1",
            [trace_id],
            |r| {
                Ok((
                    r.get::<_, Option<String>>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, bool>(2)?,
                    r.get::<_, Option<String>>(3)?,
                ))
            },
        )
        .optional()?;
    row.map(|(model, provider, streaming, stop_reason)| {
        Ok(TraceMeta {
            model,
            provider: serde_json::from_value(serde_json::Value::String(provider))?,
            streaming,
            stop_reason,
        })
    })
    .transpose()
}

/// Updates the job only if this worker still holds it.
fn guarded(tx: &Transaction, job: &Job, set: &str, args: &[&dyn rusqlite::ToSql]) -> Result<bool> {
    let sql = format!(
        "UPDATE eval_jobs SET {set}, lease_until = NULL
          WHERE id = ?1 AND status = 'running' AND attempts = ?2"
    );
    let mut all: Vec<&dyn rusqlite::ToSql> = vec![&job.id, &job.attempts];
    all.extend_from_slice(args);
    Ok(tx.execute(&sql, all.as_slice())? == 1)
}

/// Inserts the score and marks the job done in one transaction. Returns
/// `false`, writing nothing, when the lease was lost.
///
/// # Errors
///
/// Database error.
pub fn complete(conn: &mut Connection, job: &Job, s: &Score, now_ms: i64) -> Result<bool> {
    let tx = conn.transaction()?;
    if !guarded(&tx, job, "status = 'done', last_error = NULL", &[])? {
        return Ok(false);
    }
    tx.execute(
        "INSERT INTO scores (trace_id, name, data_type, value_bool, value_label, value_num,
                             confidence, distribution, source, evaluator_version, judge_model,
                             judge_latency_ms, judge_input_tokens, judge_cost_usd,
                             input_truncated, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'judge', ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
        params![
            job.trace_id,
            job.evaluator,
            s.data_type,
            s.value_bool,
            s.value_label,
            s.value_num,
            s.confidence,
            s.distribution,
            job.evaluator_version,
            s.judge_model,
            s.judge_latency_ms,
            s.judge_input_tokens,
            s.judge_cost_usd,
            s.input_truncated,
            now_ms,
        ],
    )?;
    tx.commit()?;
    Ok(true)
}

/// Back to `pending` after `delay`, or `failed` once `max_attempts` are used.
/// Returns whether this call failed the job for good.
///
/// # Errors
///
/// Database error.
pub fn retry(
    conn: &mut Connection,
    job: &Job,
    max_attempts: u32,
    delay: Duration,
    error: &str,
    now_ms: i64,
) -> Result<bool> {
    let failed = job.attempts >= max_attempts;
    let tx = conn.transaction()?;
    let status = if failed { "failed" } else { "pending" };
    let applied = guarded(
        &tx,
        job,
        "status = ?3, not_before = ?4, last_error = ?5",
        &[&status, &now_ms.saturating_add(ms(delay)), &error],
    )?;
    tx.commit()?;
    Ok(failed && applied)
}

/// `failed` without further attempts.
///
/// # Errors
///
/// Database error.
pub fn fail(conn: &mut Connection, job: &Job, error: &str) -> Result<()> {
    let tx = conn.transaction()?;
    guarded(&tx, job, "status = 'failed', last_error = ?3", &[&error])?;
    tx.commit()?;
    Ok(())
}

/// Back to `pending` at once, the attempt not counted: used on shutdown.
///
/// # Errors
///
/// Database error.
pub fn release(conn: &mut Connection, job: &Job, now_ms: i64) -> Result<()> {
    let tx = conn.transaction()?;
    guarded(
        &tx,
        job,
        "status = 'pending', attempts = attempts - 1, not_before = ?3",
        &[&now_ms],
    )?;
    tx.commit()?;
    Ok(())
}

/// Job count per status.
///
/// # Errors
///
/// Database error.
pub fn counts(conn: &Connection) -> Result<Vec<(String, i64)>> {
    let mut stmt = conn.prepare_cached("SELECT status, count(*) FROM eval_jobs GROUP BY status")?;
    let rows = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<Result<_, _>>()?;
    Ok(rows)
}

/// Writes each evaluator's pass rule for `score_verdicts`. Rows of
/// evaluators no longer configured stay.
///
/// # Errors
///
/// Database error.
pub fn register(conn: &mut Connection, evaluators: &[Evaluator], now_ms: i64) -> Result<()> {
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare_cached(
            "INSERT INTO evaluators (name, kind, pass, min_confidence, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT (name) DO UPDATE SET kind = excluded.kind, pass = excluded.pass,
                 min_confidence = excluded.min_confidence, updated_at = excluded.updated_at",
        )?;
        for e in evaluators {
            let pass = e.pass.as_ref().map(serde_json::to_string).transpose()?;
            stmt.execute(params![
                e.name,
                e.question.kind(),
                pass,
                e.min_confidence,
                now_ms
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Exponential from 1 s, capped at 5 min, with jitter in `[50%, 100%]`;
/// never shorter than the server's `retry_after`.
#[must_use]
pub fn backoff(job: &Job, retry_after: Option<Duration>) -> Duration {
    let exp = BACKOFF_BASE
        .saturating_mul(1 << job.attempts.saturating_sub(1).min(16))
        .min(BACKOFF_MAX);
    let digest =
        Sha256::digest([job.id.to_be_bytes(), i64::from(job.attempts).to_be_bytes()].concat());
    let jitter = 0.5 + f64::from(digest[0]) / 510.0;
    exp.mul_f64(jitter).max(retry_after.unwrap_or_default())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn job(attempts: u32) -> Job {
        Job {
            id: 7,
            trace_id: vec![1],
            evaluator: "e".into(),
            evaluator_version: "v".into(),
            attempts,
        }
    }

    #[test]
    fn backoff_grows_caps_and_honors_retry_after() {
        let b1 = backoff(&job(1), None);
        assert!(
            (Duration::from_millis(500)..=Duration::from_secs(1)).contains(&b1),
            "{b1:?}"
        );
        let b4 = backoff(&job(4), None);
        assert!(
            (Duration::from_secs(4)..=Duration::from_secs(8)).contains(&b4),
            "{b4:?}"
        );
        assert!(backoff(&job(30), None) <= BACKOFF_MAX);
        assert_eq!(
            backoff(&job(1), Some(Duration::from_secs(2))),
            Duration::from_secs(2)
        );
    }
}
