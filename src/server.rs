//! Proxy and admin listeners, plus graceful shutdown.

use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::Router;
use axum::http::header;
use axum::middleware;
use axum::routing::get;
use axum_server::Handle;
use axum_server::tls_rustls::RustlsConfig;

use crate::gateway::Gateway;
use crate::modules::Scrape;
use crate::{ingress, proxy};

/// No gateway routes on this port, so no upstream name is reserved.
pub fn proxy_router(gw: Arc<Gateway>) -> Router {
    Router::new()
        .fallback(proxy::forward)
        .layer(middleware::from_fn_with_state(gw.clone(), ingress::check))
        .with_state(gw)
}

/// `/metrics` exists only with the `[prometheus]` module.
pub fn admin_router(metrics: Option<Scrape>) -> Router {
    let router = Router::new().route("/healthz", get(|| async { "ok" }));
    match metrics {
        Some(scrape) => router.route(
            "/metrics",
            get(move || async move {
                (
                    [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
                    scrape(),
                )
            }),
        ),
        None => router,
    }
}

#[derive(Clone, Default)]
pub struct Handles {
    proxy: Handle<SocketAddr>,
    admin: Handle<SocketAddr>,
}

impl Handles {
    pub fn graceful_shutdown(&self, grace: Duration) {
        self.proxy.graceful_shutdown(Some(grace));
        self.admin.graceful_shutdown(Some(grace));
    }
}

/// Listeners are bound by the caller, so tests can bind port 0.
///
/// # Errors
///
/// A listener fails to register with tokio or stops with an I/O error.
pub async fn serve(
    gw: Arc<Gateway>,
    proxy_listener: TcpListener,
    admin_listener: TcpListener,
    tls: Option<RustlsConfig>,
    handles: Handles,
) -> Result<()> {
    proxy_listener.set_nonblocking(true)?;
    admin_listener.set_nonblocking(true)?;

    let admin_app = admin_router(gw.metrics.clone()).into_make_service();
    let proxy_app = proxy_router(gw).into_make_service_with_connect_info::<SocketAddr>();
    let proxy = async {
        match tls {
            Some(tls) => {
                axum_server::tls_rustls::from_tcp_rustls(proxy_listener, tls)?
                    .handle(handles.proxy.clone())
                    .serve(proxy_app)
                    .await
            }
            None => {
                axum_server::from_tcp(proxy_listener)?
                    .handle(handles.proxy.clone())
                    .serve(proxy_app)
                    .await
            }
        }
    };
    let admin = async {
        axum_server::from_tcp(admin_listener)?
            .handle(handles.admin.clone())
            .serve(admin_app)
            .await
    };

    let (proxy, admin) = tokio::join!(proxy, admin);
    proxy.context("proxy listener")?;
    admin.context("admin listener")?;
    Ok(())
}

/// Wait for SIGTERM or SIGINT.
///
/// # Errors
///
/// Signal handlers cannot be installed.
pub async fn shutdown_signal() -> Result<()> {
    use tokio::signal::unix::{SignalKind, signal};
    let mut term = signal(SignalKind::terminate())?;
    let mut int = signal(SignalKind::interrupt())?;
    tokio::select! {
        _ = term.recv() => {}
        _ = int.recv() => {}
    }
    Ok(())
}

/// Process-wide rustls provider for the TLS listener and upstream client.
/// Idempotent.
pub fn install_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}
