//! Content-addressed blobs: sha256 of the stored bytes, zstd-compressed.

use anyhow::{Result, bail};
use rusqlite::{Connection, params};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

const CODEC: &str = "zstd";
const LEVEL: i32 = 3;

/// Keys sorted at every level and `cache_control` dropped: harnesses move
/// cache breakpoints every turn, which would make each resent message new.
#[must_use]
pub fn canonical(v: &Value) -> Value {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<_> = m.keys().filter(|k| *k != "cache_control").collect();
            keys.sort();
            let out: Map<String, Value> = keys
                .into_iter()
                .map(|k| (k.clone(), canonical(&m[k])))
                .collect();
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

/// Stores `v` in canonical form; returns its hash.
pub fn put_json(conn: &Connection, v: &Value) -> Result<Vec<u8>> {
    put(conn, &serde_json::to_vec(&canonical(v))?)
}

/// Compresses only when the hash is new.
pub fn put(conn: &Connection, bytes: &[u8]) -> Result<Vec<u8>> {
    let hash = Sha256::digest(bytes).to_vec();
    let exists = conn
        .prepare_cached("SELECT 1 FROM blobs WHERE hash = ?1")?
        .exists([&hash])?;
    if !exists {
        let compressed = zstd::bulk::compress(bytes, LEVEL)?;
        conn.prepare_cached(
            "INSERT OR IGNORE INTO blobs (hash, codec, bytes) VALUES (?1, ?2, ?3)",
        )?
        .execute(params![hash, CODEC, compressed])?;
    }
    Ok(hash)
}

pub fn decode(codec: &str, bytes: &[u8]) -> Result<Vec<u8>> {
    match codec {
        CODEC => Ok(zstd::stream::decode_all(bytes)?),
        other => bail!("unknown blob codec {other:?}"),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn canonical_sorts_keys_and_drops_cache_control() {
        let a = json!({"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]});
        let b = json!({"content": [{"text": "hi", "type": "text"}], "role": "user"});
        let ca = serde_json::to_string(&canonical(&a)).unwrap();
        assert_eq!(ca, serde_json::to_string(&canonical(&b)).unwrap());
        assert_eq!(
            ca,
            r#"{"content":[{"text":"hi","type":"text"}],"role":"user"}"#
        );
    }

    #[test]
    fn put_dedups_and_decodes() {
        let dir = tempfile::tempdir().unwrap();
        let conn = super::super::schema::open_writer(&dir.path().join("g.db")).unwrap();
        let h1 = put_json(&conn, &json!({"b": 1, "a": 2})).unwrap();
        let h2 = put_json(&conn, &json!({"a": 2, "b": 1})).unwrap();
        assert_eq!(h1, h2);
        let (n, codec, bytes): (u64, String, Vec<u8>) = conn
            .query_row("SELECT count(*), codec, bytes FROM blobs", [], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .unwrap();
        assert_eq!(n, 1);
        assert_eq!(decode(&codec, &bytes).unwrap(), br#"{"a":2,"b":1}"#);
        assert!(decode("lz4", &bytes).is_err());
    }
}
