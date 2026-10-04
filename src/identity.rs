//! Who made a request: a pseudonymous caller id plus client-supplied session
//! and tags. Read from `x-gateway-*` headers, which are never forwarded.

use std::collections::BTreeMap;
use std::fmt;

use axum::http::HeaderMap;
use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;

pub const IDENTITY_SECRET_ENV: &str = "IG_IDENTITY_SECRET";

const SESSION_HEADER: &str = "x-gateway-session";
/// Sent by Claude Code on every request; forwarded upstream unchanged.
const CLAUDE_CODE_SESSION_HEADER: &str = "x-claude-code-session-id";
const TAG_PREFIX: &str = "x-gateway-tag-";
const CAPTURE_HEADER: &str = "x-gateway-capture";

/// Generate with `openssl rand -hex 32`. Changing it changes every `caller_id`.
#[derive(Clone)]
pub struct IdentitySecret(Vec<u8>);

impl IdentitySecret {
    /// `None` for an empty secret.
    #[must_use]
    pub fn new(secret: &str) -> Option<Self> {
        (!secret.is_empty()).then(|| Self(secret.as_bytes().to_vec()))
    }

    /// First 16 hex chars of HMAC-SHA256(secret, api key).
    #[must_use]
    pub fn caller_id(&self, api_key: &str) -> String {
        let Ok(mut mac) = Hmac::<Sha256>::new_from_slice(&self.0) else {
            unreachable!("HMAC takes keys of any length")
        };
        mac.update(api_key.as_bytes());
        let mut id = hex::encode(mac.finalize().into_bytes());
        id.truncate(16);
        id
    }
}

impl fmt::Debug for IdentitySecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("IdentitySecret(<redacted>)")
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Principal {
    /// `None` without an API key or without a secret.
    pub caller_id: Option<String>,
    pub session_id: Option<String>,
    pub tags: BTreeMap<String, String>,
}

impl Principal {
    /// Header values that are not visible ASCII are ignored.
    #[must_use]
    pub fn from_headers(
        headers: &HeaderMap,
        api_key: Option<&str>,
        secret: Option<&IdentitySecret>,
    ) -> Self {
        let caller_id = secret.zip(api_key).map(|(s, k)| s.caller_id(k));
        let session_id = [SESSION_HEADER, CLAUDE_CODE_SESSION_HEADER]
            .into_iter()
            .find_map(|name| {
                headers
                    .get(name)
                    .and_then(|v| v.to_str().ok())
                    .filter(|v| !v.is_empty())
            })
            .map(str::to_owned);
        let tags = headers
            .iter()
            .filter_map(|(name, value)| {
                let key = name.as_str().strip_prefix(TAG_PREFIX)?;
                Some((key.to_owned(), value.to_str().ok()?.to_owned()))
            })
            .filter(|(k, _)| !k.is_empty())
            .collect();
        Self {
            caller_id,
            session_id,
            tags,
        }
    }
}

/// `x-gateway-capture: 0|1`; other values are ignored.
#[must_use]
pub fn capture_header(headers: &HeaderMap) -> Option<bool> {
    match headers.get(CAPTURE_HEADER)?.as_bytes() {
        b"1" => Some(true),
        b"0" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use axum::http::HeaderValue;

    use super::*;

    fn secret(s: &str) -> IdentitySecret {
        IdentitySecret::new(s).unwrap()
    }

    #[test]
    fn caller_id_is_stable_and_per_key() {
        let s = secret("k1");
        assert_eq!(s.caller_id("sk-a"), s.caller_id("sk-a"));
        assert_eq!(s.caller_id("sk-a").len(), 16);
        assert_ne!(s.caller_id("sk-a"), s.caller_id("sk-b"));
        assert_ne!(s.caller_id("sk-a"), secret("k2").caller_id("sk-a"));
    }

    #[test]
    fn caller_id_matches_reference_hmac() {
        // python3 -c "import hmac,hashlib;print(hmac.new(b'k1',b'sk-a',hashlib.sha256).hexdigest())"
        assert_eq!(secret("k1").caller_id("sk-a"), "360a4de4862cf4cc");
    }

    #[test]
    fn empty_secret_is_none() {
        assert!(IdentitySecret::new("").is_none());
        assert_eq!(
            format!("{:?}", secret("hunter2")),
            "IdentitySecret(<redacted>)"
        );
    }

    #[test]
    fn reads_session_and_tags() {
        let mut h = HeaderMap::new();
        h.insert("x-gateway-session", HeaderValue::from_static("run-1"));
        h.insert("x-gateway-tag-project", HeaderValue::from_static("ig"));
        h.insert("x-gateway-tag-exp", HeaderValue::from_static("a"));
        h.insert("x-gateway-tag-", HeaderValue::from_static("nameless"));
        h.insert("x-other", HeaderValue::from_static("x"));
        let p = Principal::from_headers(&h, Some("sk-a"), Some(&secret("k1")));
        assert_eq!(p.caller_id.as_deref(), Some("360a4de4862cf4cc"));
        assert_eq!(p.session_id.as_deref(), Some("run-1"));
        assert_eq!(
            p.tags.into_iter().collect::<Vec<_>>(),
            [("exp".into(), "a".into()), ("project".into(), "ig".into())]
        );
    }

    #[test]
    fn claude_code_session_is_the_fallback() {
        let mut h = HeaderMap::new();
        h.insert("x-claude-code-session-id", HeaderValue::from_static("cc-1"));
        assert_eq!(
            Principal::from_headers(&h, None, None)
                .session_id
                .as_deref(),
            Some("cc-1")
        );
        h.insert("x-gateway-session", HeaderValue::from_static(""));
        assert_eq!(
            Principal::from_headers(&h, None, None)
                .session_id
                .as_deref(),
            Some("cc-1")
        );
        h.insert("x-gateway-session", HeaderValue::from_static("run-1"));
        assert_eq!(
            Principal::from_headers(&h, None, None)
                .session_id
                .as_deref(),
            Some("run-1")
        );
    }

    #[test]
    fn no_secret_or_no_key_means_no_caller() {
        let h = HeaderMap::new();
        assert_eq!(
            Principal::from_headers(&h, Some("sk"), None).caller_id,
            None
        );
        assert_eq!(
            Principal::from_headers(&h, None, Some(&secret("k"))).caller_id,
            None
        );
    }

    #[test]
    fn capture_header_is_zero_or_one() {
        let get = |v: &'static str| {
            let mut h = HeaderMap::new();
            h.insert("x-gateway-capture", HeaderValue::from_static(v));
            capture_header(&h)
        };
        assert_eq!(get("1"), Some(true));
        assert_eq!(get("0"), Some(false));
        assert_eq!(get("yes"), None);
        assert_eq!(capture_header(&HeaderMap::new()), None);
    }
}
