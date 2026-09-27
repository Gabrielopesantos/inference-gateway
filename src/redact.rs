//! Header redaction for logs. Keys are never logged.

use std::fmt;

use axum::http::HeaderMap;
use axum::http::header::HeaderName;

const REDACTED: &str = "<redacted>";

/// Headers whose values are credentials.
const SECRET_HEADERS: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "cookie",
    "set-cookie",
    "x-gateway-token",
];

#[must_use]
pub fn is_secret(name: &HeaderName) -> bool {
    SECRET_HEADERS.contains(&name.as_str())
}

/// Borrowing view of a header map that prints secret values as `<redacted>`.
pub struct RedactedHeaders<'a>(&'a HeaderMap);

#[must_use]
pub fn headers(map: &HeaderMap) -> RedactedHeaders<'_> {
    RedactedHeaders(map)
}

impl fmt::Debug for RedactedHeaders<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.0.iter().map(|(name, value)| {
                let shown = if is_secret(name) {
                    REDACTED
                } else {
                    value.to_str().unwrap_or("<binary>")
                };
                (name.as_str(), shown)
            }))
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use axum::http::HeaderValue;

    use super::*;

    #[test]
    fn secrets_are_redacted_others_kept() {
        let mut map = HeaderMap::new();
        map.insert("x-api-key", HeaderValue::from_static("sk-ant-secret"));
        map.insert(
            "authorization",
            HeaderValue::from_static("Bearer sk-secret"),
        );
        map.insert("x-gateway-token", HeaderValue::from_static("tok"));
        map.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        let shown = format!("{:?}", headers(&map));
        assert!(!shown.contains("secret"), "{shown}");
        assert!(!shown.contains("tok\""), "{shown}");
        assert!(shown.contains("2023-06-01"), "{shown}");
        assert_eq!(shown.matches(REDACTED).count(), 3, "{shown}");
    }
}
