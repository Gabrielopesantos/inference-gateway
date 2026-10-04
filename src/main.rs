//! `inference-gateway` binary: parse flags, load the config file, serve.

use std::net::TcpListener;

use anyhow::{Context, Result};
use axum_server::tls_rustls::RustlsConfig;
use clap::Parser;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use inference_gateway::config::Config;
use inference_gateway::config::cli::{Cli, Command, LogFormat, ServeArgs};
use inference_gateway::gateway::{Gateway, Secrets};
use inference_gateway::identity::IDENTITY_SECRET_ENV;
use inference_gateway::modules::{self, BuildCtx, Dispatcher};
use inference_gateway::server::{self, Handles};

#[tokio::main]
async fn main() -> Result<()> {
    let command = Cli::parse().into_command().unwrap_or_else(|e| e.exit());
    match command {
        Command::Serve(args) => run_serve(args).await,
    }
}

fn init_tracing(args: &ServeArgs) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&args.log_level));
    let builder = tracing_subscriber::fmt().with_env_filter(filter);
    match args.log_format {
        LogFormat::Text => builder.init(),
        LogFormat::Json => builder.json().init(),
    }
}

async fn run_serve(args: ServeArgs) -> Result<()> {
    init_tracing(&args);
    server::install_crypto_provider();

    let config = Config::load(&args.config)?;
    let upstreams: Vec<String> = config
        .upstreams
        .iter()
        .map(|u| format!("{} -> {} ({})", u.name, u.base_url, u.provider))
        .collect();
    let mut ctx = BuildCtx::default();
    let dispatcher = Dispatcher::start(
        modules::build(&config, &mut ctx)?,
        &ctx,
        Dispatcher::QUEUE_CAPACITY,
    );
    let gw = Gateway::build(config, &args, Secrets::from_env(), &dispatcher)?;
    if gw.identity.is_none() {
        info!("{IDENTITY_SECRET_ENV} unset; records carry no caller_id");
    }

    if gw.ingress.allow_cidrs.is_empty()
        && gw.ingress.token.is_none()
        && !args.listen.ip().is_loopback()
    {
        warn!(
            listen = %args.listen,
            "proxy listens on a non-loopback address with no allow_cidrs and no token; anyone who can reach it can relay through it"
        );
    }

    let tls = match (&args.tls_cert, &args.tls_key) {
        (Some(cert), Some(key)) => Some(
            RustlsConfig::from_pem_file(cert, key)
                .await
                .with_context(|| {
                    format!(
                        "loading TLS cert {} / key {}",
                        cert.display(),
                        key.display()
                    )
                })?,
        ),
        _ => None,
    };

    let proxy_listener =
        TcpListener::bind(args.listen).with_context(|| format!("binding {}", args.listen))?;
    let admin_listener = TcpListener::bind(args.admin_listen)
        .with_context(|| format!("binding {}", args.admin_listen))?;
    info!(
        listen = %args.listen,
        admin = %args.admin_listen,
        tls = tls.is_some(),
        ?upstreams,
        "inference-gateway starting"
    );

    let handles = Handles::default();
    let mut serving = tokio::spawn(server::serve(
        gw,
        proxy_listener,
        admin_listener,
        tls,
        handles.clone(),
    ));

    tokio::select! {
        res = &mut serving => res.context("server task")??,
        res = server::shutdown_signal() => {
            res?;
            info!(grace = ?args.shutdown_timeout, "shutdown signal received; draining in-flight requests");
            handles.graceful_shutdown(args.shutdown_timeout);
            serving.await.context("server task")??;
        }
    }
    dispatcher.shutdown(args.shutdown_timeout).await;
    info!("shutdown complete");
    Ok(())
}
