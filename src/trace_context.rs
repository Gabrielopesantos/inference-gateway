//! W3C `traceparent` from the client, so gateway spans join the caller's
//! trace. The header is forwarded upstream unchanged.

use axum::http::HeaderMap;

const TRACEPARENT: &str = "traceparent";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceParent {
    pub trace_id: [u8; 16],
    pub parent_span_id: [u8; 8],
    pub sampled: bool,
}

impl TraceParent {
    /// `None` when absent or invalid.
    #[must_use]
    pub fn from_headers(headers: &HeaderMap) -> Option<Self> {
        Self::parse(headers.get(TRACEPARENT)?.to_str().ok()?)
    }

    /// `version-traceid-parentid-flags`, lowercase hex. Later versions may
    /// append fields after the flags.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        let mut fields = value.trim().split('-');
        let version: [u8; 1] = hex_field(fields.next()?)?;
        let trace_id: [u8; 16] = hex_field(fields.next()?)?;
        let parent_span_id: [u8; 8] = hex_field(fields.next()?)?;
        let flags: [u8; 1] = hex_field(fields.next()?)?;
        let rest = fields.next();
        let valid_version = match version[0] {
            0 => rest.is_none(),
            0xff => false,
            _ => true,
        };
        let valid_ids = trace_id != [0; 16] && parent_span_id != [0; 8];
        (valid_version && valid_ids).then_some(Self {
            trace_id,
            parent_span_id,
            sampled: flags[0] & 1 == 1,
        })
    }
}

fn hex_field<const N: usize>(s: &str) -> Option<[u8; N]> {
    if s.len() != 2 * N || s.bytes().any(|b| b.is_ascii_uppercase()) {
        return None;
    }
    let mut out = [0; N];
    hex::decode_to_slice(s, &mut out).ok()?;
    Some(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const VALID: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

    #[test]
    fn parses_valid_header() {
        let tp = TraceParent::parse(VALID).unwrap();
        assert_eq!(hex::encode(tp.trace_id), "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(hex::encode(tp.parent_span_id), "00f067aa0ba902b7");
        assert!(tp.sampled);
    }

    #[test]
    fn sampled_is_bit_zero() {
        let tp = TraceParent::parse(&VALID.replace("-01", "-00")).unwrap();
        assert!(!tp.sampled);
        let tp = TraceParent::parse(&VALID.replace("-01", "-03")).unwrap();
        assert!(tp.sampled);
    }

    #[test]
    fn rejects_invalid() {
        for bad in [
            "",
            "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
            "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
            "0x-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        ] {
            assert_eq!(TraceParent::parse(bad), None, "{bad}");
        }
    }

    #[test]
    fn later_versions_may_append_fields() {
        let v = "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-future";
        assert!(TraceParent::parse(v).is_some());
    }

    #[test]
    fn reads_header() {
        let mut headers = HeaderMap::new();
        headers.insert(TRACEPARENT, VALID.parse().unwrap());
        assert!(TraceParent::from_headers(&headers).is_some());
        assert_eq!(TraceParent::from_headers(&HeaderMap::new()), None);
    }
}
