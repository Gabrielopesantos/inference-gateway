//! Deletes captured content older than the retention window. Trace rows are
//! kept. Runs on the writer thread, triggered by a timer task.

use std::time::Duration;

use anyhow::Result;
use rusqlite::Connection;
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;
use tracing::debug;

use super::writer::Msg;

const INTERVAL: Duration = Duration::from_hours(1);

/// First tick is immediate, so a restart catches up at once.
pub async fn run(tx: mpsc::Sender<Msg>, max_age: Duration) {
    let mut every = tokio::time::interval(INTERVAL);
    every.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        every.tick().await;
        if tx.send(Msg::Retain(max_age)).await.is_err() {
            return;
        }
    }
}

/// Content rows of traces started before `cutoff_ms`, then blobs no content
/// row references. Driven from the content tables, which retention keeps
/// small; trace rows grow forever.
pub fn retain(conn: &mut Connection, cutoff_ms: i64) -> Result<()> {
    let tx = conn.transaction()?;
    let mut deleted = 0;
    for table in ["trace_inputs", "trace_params", "trace_outputs"] {
        let sql = format!(
            "DELETE FROM {table} WHERE EXISTS
               (SELECT 1 FROM traces t WHERE t.id = {table}.trace_id AND t.started_at < ?1)"
        );
        deleted += tx.execute(&sql, [cutoff_ms])?;
    }
    let blobs = tx.execute(
        "DELETE FROM blobs
         WHERE NOT EXISTS (SELECT 1 FROM trace_inputs WHERE blob_hash = blobs.hash)
           AND NOT EXISTS (SELECT 1 FROM trace_outputs WHERE blob_hash = blobs.hash)",
        [],
    )?;
    tx.commit()?;
    // Returns freed pages to the filesystem; needs auto_vacuum = INCREMENTAL.
    conn.execute_batch("PRAGMA incremental_vacuum")?;
    debug!(rows = deleted, blobs, "retention pass done");
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use rusqlite::params;

    use super::super::schema;
    use super::*;

    fn trace(conn: &Connection, id: u8, started_at: i64) {
        conn.execute(
            "INSERT INTO traces (id, started_at, upstream, provider, method, path, streaming,
                                 tags, outcome, duration_ms, chunk_count, captured)
             VALUES (?1, ?2, 'a', 'anthropic', 'POST', '/v1/messages', 1, '{}', 'ok', 1, 1, 1)",
            params![[id], started_at],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO trace_params (trace_id, json) VALUES (?1, '{}')",
            [[id]],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO trace_outputs (trace_id, blob_hash, truncated) VALUES (?1, 'raw', 0)",
            [[id]],
        )
        .unwrap();
    }

    fn input(conn: &Connection, id: u8, ord: u32, blob: &str) {
        conn.execute(
            "INSERT INTO trace_inputs (trace_id, ord, kind, role, blob_hash)
             VALUES (?1, ?2, 'message', 'user', ?3)",
            params![[id], ord, blob],
        )
        .unwrap();
    }

    fn count(conn: &Connection, sql: &str) -> i64 {
        conn.query_row(sql, [], |r| r.get(0)).unwrap()
    }

    #[test]
    fn deletes_expired_content_and_orphan_blobs_only() {
        let dir = tempfile::tempdir().unwrap();
        let mut conn = schema::open_writer(&dir.path().join("g.db")).unwrap();
        for hash in ["shared", "old_only", "new_only", "raw"] {
            conn.execute(
                "INSERT INTO blobs (hash, codec, bytes) VALUES (?1, 'zstd', x'00')",
                [hash],
            )
            .unwrap();
        }
        trace(&conn, 1, 1_000);
        input(&conn, 1, 0, "shared");
        input(&conn, 1, 1, "old_only");
        trace(&conn, 2, 5_000);
        input(&conn, 2, 0, "shared");
        input(&conn, 2, 1, "new_only");

        retain(&mut conn, 3_000).unwrap();

        assert_eq!(count(&conn, "SELECT count(*) FROM traces"), 2);
        for table in ["trace_inputs", "trace_params", "trace_outputs"] {
            let old = count(
                &conn,
                &format!("SELECT count(*) FROM {table} WHERE trace_id = x'01'"),
            );
            let new = count(
                &conn,
                &format!("SELECT count(*) FROM {table} WHERE trace_id = x'02'"),
            );
            assert_eq!((old, new == 0), (0, false), "{table}");
        }
        let blobs: Vec<String> = conn
            .prepare("SELECT hash FROM blobs ORDER BY hash")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(blobs, ["new_only", "raw", "shared"]);
    }
}
