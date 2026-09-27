//! Errors the gateway itself answers with (as opposed to upstream errors,
//! which pass through unchanged).

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("unknown upstream {0:?}")]
    UnknownUpstream(String),
    #[error("request body exceeds {limit} bytes")]
    BodyTooLarge { limit: u64 },
    #[error("reading request body: {0}")]
    BadRequestBody(String),
    #[error("invalid request path: {0}")]
    InvalidPath(String),
    #[error("client address not allowed")]
    Forbidden,
    #[error("missing or invalid x-gateway-token")]
    Unauthorized,
    #[error("upstream unreachable: {0}")]
    UpstreamUnreachable(String),
    #[error("upstream timed out: {0}")]
    UpstreamTimeout(String),
}

impl GatewayError {
    #[must_use]
    pub fn status(&self) -> StatusCode {
        match self {
            Self::UnknownUpstream(_) => StatusCode::NOT_FOUND,
            Self::BodyTooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
            Self::BadRequestBody(_) | Self::InvalidPath(_) => StatusCode::BAD_REQUEST,
            Self::Forbidden => StatusCode::FORBIDDEN,
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::UpstreamUnreachable(_) => StatusCode::BAD_GATEWAY,
            Self::UpstreamTimeout(_) => StatusCode::GATEWAY_TIMEOUT,
        }
    }

    /// Prefixed so clients can tell gateway errors from provider errors.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::UnknownUpstream(_) => "gateway_unknown_upstream",
            Self::BodyTooLarge { .. } => "gateway_body_too_large",
            Self::BadRequestBody(_) => "gateway_bad_request_body",
            Self::InvalidPath(_) => "gateway_invalid_path",
            Self::Forbidden => "gateway_forbidden",
            Self::Unauthorized => "gateway_unauthorized",
            Self::UpstreamUnreachable(_) => "gateway_upstream_unreachable",
            Self::UpstreamTimeout(_) => "gateway_upstream_timeout",
        }
    }
}

/// Body shaped like Anthropic's error envelope. `OpenAI` clients read
/// `error.message` and `error.type` from it too.
impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let body = json!({
            "type": "error",
            "error": { "type": self.kind(), "message": self.to_string() },
        });
        (self.status(), Json(body)).into_response()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use http_body_util::BodyExt;

    use super::*;

    #[tokio::test]
    async fn renders_json_envelope() {
        let resp = GatewayError::UnknownUpstream("nope".into()).into_response();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "gateway_unknown_upstream");
        assert_eq!(v["error"]["message"], r#"unknown upstream "nope""#);
    }
}
