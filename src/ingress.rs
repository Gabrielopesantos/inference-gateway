//! Proxy listener access control: client CIDR allowlist and optional
//! `x-gateway-token`. Without them a public gateway is an open relay.
//!
//! `X-Forwarded-For` is not trusted, so behind a load balancer the allowlist
//! sees the balancer's address.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::{ConnectInfo, Request, State};
use axum::http::HeaderValue;
use axum::middleware::Next;
use axum::response::Response;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::error::GatewayError;
use crate::gateway::Gateway;

pub const TOKEN_HEADER: &str = "x-gateway-token";

/// # Errors
///
/// Peer outside `allow_cidrs`, or required token missing or wrong.
pub async fn check(
    State(gw): State<Arc<Gateway>>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    req: Request,
    next: Next,
) -> Result<Response, GatewayError> {
    let policy = &gw.ingress;

    // IPv4 clients on a dual-stack socket show up as `::ffff:a.b.c.d`.
    let ip = peer.ip().to_canonical();
    if !policy.allow_cidrs.is_empty() && !policy.allow_cidrs.iter().any(|net| net.contains(&ip)) {
        return Err(GatewayError::Forbidden);
    }

    if let Some(expected) = &policy.token {
        let given = req
            .headers()
            .get(TOKEN_HEADER)
            .map(HeaderValue::as_bytes)
            .unwrap_or_default();
        // Digests have equal length, so the compare leaks no length either.
        let eq = Sha256::digest(given)[..].ct_eq(&Sha256::digest(expected)[..]);
        if !bool::from(eq) {
            return Err(GatewayError::Unauthorized);
        }
    }

    Ok(next.run(req).await)
}
