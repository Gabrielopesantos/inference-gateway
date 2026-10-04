//! Reads captured content back, for modules that work on stored traces.

use anyhow::Result;
use rusqlite::OptionalExtension;
use serde_json::Value;
use uuid::Uuid;

use super::{Store, blobs};
use crate::providers::RequestParts;

/// Captured content of one trace, as stored.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceContent {
    /// Canonical JSON: keys sorted, `cache_control` removed.
    pub request: RequestParts,
    /// Empty when no response arrived.
    pub raw_response: Vec<u8>,
    pub output_text: Option<String>,
    pub last_user_text: Option<String>,
    pub truncated: bool,
}

impl Store {
    /// `None` when the trace was not captured or its content expired.
    ///
    /// # Errors
    ///
    /// Database or blob decoding error.
    pub fn load_trace_content(&self, id: Uuid) -> Result<Option<TraceContent>> {
        let conn = self.reader()?;
        let id = id.as_bytes().as_slice();
        let output = conn
            .query_row(
                "SELECT o.output_text, o.last_user_text, o.truncated, b.codec, b.bytes
                 FROM trace_outputs o LEFT JOIN blobs b ON b.hash = o.blob_hash
                 WHERE o.trace_id = ?1",
                [id],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get::<_, Option<String>>(3)?,
                        r.get::<_, Option<Vec<u8>>>(4)?,
                    ))
                },
            )
            .optional()?;
        let Some((output_text, last_user_text, truncated, codec, bytes)) = output else {
            return Ok(None);
        };
        let raw_response = match (codec, bytes) {
            (Some(codec), Some(bytes)) => blobs::decode(&codec, &bytes)?,
            _ => Vec::new(),
        };

        let mut request = RequestParts::default();
        let mut stmt = conn.prepare(
            "SELECT i.kind, b.codec, b.bytes FROM trace_inputs i
             JOIN blobs b ON b.hash = i.blob_hash
             WHERE i.trace_id = ?1 ORDER BY i.ord",
        )?;
        let mut rows = stmt.query([id])?;
        while let Some(r) = rows.next()? {
            let kind: String = r.get(0)?;
            let json = blobs::decode(&r.get::<_, String>(1)?, &r.get::<_, Vec<u8>>(2)?)?;
            let value: Value = serde_json::from_slice(&json)?;
            match kind.as_str() {
                "system" => request.system = Some(value),
                "tools" => request.tools = Some(value),
                _ => request.messages.push(value),
            }
        }
        if let Some(json) = conn
            .query_row(
                "SELECT json FROM trace_params WHERE trace_id = ?1",
                [id],
                |r| r.get::<_, String>(0),
            )
            .optional()?
        {
            request.params = serde_json::from_str(&json)?;
        }
        Ok(Some(TraceContent {
            request,
            raw_response,
            output_text,
            last_user_text,
            truncated,
        }))
    }
}
