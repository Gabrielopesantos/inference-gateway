//! Connection setup and embedded migrations, tracked in `user_version`.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OpenFlags};

/// Applied in order; entry `n` moves `user_version` from `n` to `n + 1`.
/// Never edit a released entry: append a new one.
const MIGRATIONS: &[&str] = &[V1, V2, V3];

/// Times are unix milliseconds (INTEGER) and durations milliseconds (REAL).
/// Ids and hashes are raw bytes: `hex(id)` in SQL for display.
const V1: &str = "
CREATE TABLE traces (
    id BLOB PRIMARY KEY,
    started_at INTEGER NOT NULL,
    upstream TEXT NOT NULL,
    provider TEXT NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    operation TEXT,
    requested_model TEXT,
    response_model TEXT,
    streaming INTEGER NOT NULL,
    caller_id TEXT,
    session_id TEXT,
    tags TEXT NOT NULL,
    http_status INTEGER,
    outcome TEXT NOT NULL,
    error_type TEXT,
    stop_reason TEXT,
    input_tokens_total INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    cache_write_1h_tokens INTEGER,
    reasoning_tokens INTEGER,
    ttfb_ms REAL,
    ttft_ms REAL,
    ttft_text_ms REAL,
    duration_ms REAL NOT NULL,
    chunk_count INTEGER NOT NULL,
    chunk_gap_mean_ms REAL,
    chunk_gap_max_ms REAL,
    output_tps REAL,
    ratelimit_requests_remaining INTEGER,
    ratelimit_tokens_remaining INTEGER,
    ratelimit_input_tokens_remaining INTEGER,
    ratelimit_output_tokens_remaining INTEGER,
    retry_after_ms REAL,
    window_5h_utilization REAL,
    window_7d_utilization REAL,
    cost_usd REAL,
    captured INTEGER NOT NULL
);
CREATE INDEX traces_started_at ON traces (started_at);
CREATE INDEX traces_session_id ON traces (session_id);
CREATE INDEX traces_response_model ON traces (response_model, started_at);

-- zstd of canonical JSON (inputs) or of raw bytes (responses).
CREATE TABLE blobs (
    hash BLOB PRIMARY KEY,
    codec TEXT NOT NULL,
    bytes BLOB NOT NULL
);

-- kind: system | tools | message. role is set for messages only.
CREATE TABLE trace_inputs (
    trace_id BLOB NOT NULL REFERENCES traces (id),
    ord INTEGER NOT NULL,
    kind TEXT NOT NULL,
    role TEXT,
    blob_hash BLOB NOT NULL REFERENCES blobs (hash),
    PRIMARY KEY (trace_id, ord)
) WITHOUT ROWID;
CREATE INDEX trace_inputs_blob ON trace_inputs (blob_hash);

CREATE TABLE trace_params (
    trace_id BLOB PRIMARY KEY REFERENCES traces (id),
    json TEXT NOT NULL
);

-- One row per captured trace; blob_hash is NULL when no response arrived.
CREATE TABLE trace_outputs (
    trace_id BLOB PRIMARY KEY REFERENCES traces (id),
    blob_hash BLOB REFERENCES blobs (hash),
    output_text TEXT,
    last_user_text TEXT,
    truncated INTEGER NOT NULL
);
CREATE INDEX trace_outputs_blob ON trace_outputs (blob_hash);

CREATE TABLE eval_jobs (
    id INTEGER PRIMARY KEY,
    trace_id BLOB NOT NULL REFERENCES traces (id),
    evaluator TEXT NOT NULL,
    evaluator_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'done', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    lease_until INTEGER,
    not_before INTEGER NOT NULL,
    last_error TEXT,
    created_at INTEGER NOT NULL
);
-- One job per trace and evaluator version: re-enqueueing is a no-op.
CREATE UNIQUE INDEX eval_jobs_once ON eval_jobs (trace_id, evaluator_version);
CREATE INDEX eval_jobs_claim ON eval_jobs (status, not_before);
CREATE INDEX eval_jobs_lease ON eval_jobs (status, lease_until);

CREATE TABLE scores (
    id INTEGER PRIMARY KEY,
    trace_id BLOB NOT NULL REFERENCES traces (id),
    name TEXT NOT NULL,
    data_type TEXT NOT NULL CHECK (data_type IN ('bool', 'categorical', 'numeric')),
    value_bool INTEGER,
    value_label TEXT,
    value_num REAL,
    confidence REAL,
    distribution TEXT,
    reason TEXT,
    source TEXT NOT NULL CHECK (source IN ('judge', 'human', 'rule')),
    evaluator_version TEXT,
    judge_model TEXT,
    judge_latency_ms REAL,
    judge_input_tokens INTEGER,
    judge_cost_usd REAL,
    input_truncated INTEGER,
    created_at INTEGER NOT NULL
);
CREATE INDEX scores_trace ON scores (trace_id);
CREATE INDEX scores_version ON scores (name, evaluator_version);
";

/// The span's trace id when the client sent a `traceparent`; NULL means `id`.
const V2: &str = "ALTER TABLE traces ADD COLUMN otel_trace_id BLOB;";

/// The pass rule of each configured evaluator, kept after it leaves the
/// config. `score_verdicts` applies it to every score of that name.
const V3: &str = "
CREATE TABLE evaluators (
    name TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('noul', 'choice', 'score')),
    -- JSON array of passing labels; NULL means no verdict.
    pass TEXT,
    min_confidence REAL,
    updated_at INTEGER NOT NULL
);

-- label: value_label, or noul as 'true' / 'false'.
-- verdict: NULL without a pass rule, else 'uncertain' below min_confidence,
-- else 'pass' or 'fail'.
CREATE VIEW score_verdicts AS
SELECT s.*,
       CASE
           WHEN e.pass IS NULL THEN NULL
           WHEN s.confidence < e.min_confidence THEN 'uncertain'
           WHEN s.label IN (SELECT value FROM json_each(e.pass)) THEN 'pass'
           ELSE 'fail'
       END AS verdict
FROM (SELECT *,
             coalesce(value_label, CASE value_bool WHEN 1 THEN 'true' WHEN 0 THEN 'false' END) AS label
        FROM scores) AS s
LEFT JOIN evaluators e ON e.name = s.name;
";

const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// The single write connection: creates the file and migrates it.
///
/// # Errors
///
/// File cannot be opened or migrated.
pub fn open_writer(path: &Path) -> Result<Connection> {
    let mut conn =
        Connection::open(path).with_context(|| format!("opening store {}", path.display()))?;
    // auto_vacuum only takes effect before the first table is created.
    conn.execute_batch(
        "PRAGMA auto_vacuum = INCREMENTAL;
         PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA foreign_keys = ON;",
    )?;
    conn.busy_timeout(BUSY_TIMEOUT)?;
    migrate(&mut conn).with_context(|| format!("migrating store {}", path.display()))?;
    Ok(conn)
}

/// # Errors
///
/// File missing or unreadable.
pub fn open_reader(path: &Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .with_context(|| format!("opening store {}", path.display()))?;
    conn.busy_timeout(BUSY_TIMEOUT)?;
    Ok(conn)
}

fn migrate(conn: &mut Connection) -> Result<()> {
    let version: usize = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
    if version > MIGRATIONS.len() {
        bail!(
            "schema version {version} is newer than this build ({})",
            MIGRATIONS.len()
        );
    }
    for (i, sql) in MIGRATIONS.iter().enumerate().skip(version) {
        let tx = conn.transaction()?;
        tx.execute_batch(sql)?;
        tx.pragma_update(None, "user_version", i + 1)?;
        tx.commit()?;
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn migrates_once_and_sets_pragmas() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.db");
        drop(open_writer(&path).unwrap());
        let conn = open_writer(&path).unwrap();
        let get = |p: &str| -> String {
            conn.pragma_query_value(None, p, |r| r.get::<_, rusqlite::types::Value>(0))
                .map(|v| format!("{v:?}"))
                .unwrap()
        };
        assert_eq!(get("user_version"), "Integer(3)");
        assert_eq!(get("journal_mode"), r#"Text("wal")"#);
        // 2 = incremental
        assert_eq!(get("auto_vacuum"), "Integer(2)");
    }

    #[test]
    fn newer_schema_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.db");
        Connection::open(&path)
            .unwrap()
            .pragma_update(None, "user_version", 99)
            .unwrap();
        let e = format!("{:#}", open_writer(&path).unwrap_err());
        assert!(e.contains("schema version 99 is newer"), "{e}");
    }
}
