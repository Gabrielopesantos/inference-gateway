//! `inference-gateway` binary: parse flags, load the config file, serve.

use std::net::TcpListener;

use anyhow::{Context, Result};
use axum_server::tls_rustls::RustlsConfig;
use clap::Parser;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use inference_gateway::config::Config;
#[cfg(feature = "eval")]
use inference_gateway::config::cli::{BackfillArgs, EvalCommand, EvalRunArgs};
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
        #[cfg(feature = "eval")]
        Command::Eval(EvalCommand::Run(args)) => run_eval(args).await,
        #[cfg(feature = "eval")]
        Command::Eval(EvalCommand::Backfill(args)) => run_backfill(&args),
        #[cfg(not(feature = "eval"))]
        Command::Eval(_) => anyhow::bail!("eval needs cargo feature `eval`"),
    }
}

fn init_tracing(level: &str, format: LogFormat) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));
    let builder = tracing_subscriber::fmt().with_env_filter(filter);
    match format {
        LogFormat::Text => builder.init(),
        LogFormat::Json => builder.json().init(),
    }
}

async fn run_serve(args: ServeArgs) -> Result<()> {
    init_tracing(&args.log_level, args.log_format);
    server::install_crypto_provider();

    let config = Config::load(&args.config)?;
    let upstreams: Vec<String> = config
        .upstreams
        .iter()
        .map(|u| format!("{} -> {} ({})", u.name, u.base_url, u.provider))
        .collect();
    let mut ctx = BuildCtx::default();
    let built = modules::build(&config, &mut ctx)?;
    let module_names: Vec<&str> = built.iter().map(|m| m.name()).collect();
    let dispatcher = Dispatcher::start(built, &ctx, Dispatcher::QUEUE_CAPACITY);
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
        modules = ?module_names,
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

/// `[store]` and `[eval]` of a config file, with the database migrated.
#[cfg(feature = "eval")]
fn eval_config(
    path: &std::path::Path,
) -> Result<(
    inference_gateway::store::Store,
    inference_gateway::config::eval::EvalConfig,
)> {
    let config = Config::load(path)?;
    let (Some(store), Some(eval)) = (config.store, config.eval) else {
        anyhow::bail!("config {} needs [store] and [eval]", path.display());
    };
    Ok((inference_gateway::store::Store::open(&store.path)?, eval))
}

#[cfg(feature = "eval")]
async fn run_eval(args: EvalRunArgs) -> Result<()> {
    use inference_gateway::eval::{EvalModule, judge_key, metrics::Metrics};
    use inference_gateway::modules::Module as _;

    init_tracing(&args.log_level, args.log_format);
    server::install_crypto_provider();
    let (store, eval) = eval_config(&args.config)?;
    let key = judge_key(&eval.judge)?;
    let evaluators: Vec<_> = eval.evaluators.iter().map(|e| e.name.as_str()).collect();
    info!(?evaluators, db = %store.path().display(), "eval worker starting");
    // No in-process writer to wake it: polls only.
    let module = EvalModule::spawn(
        &eval,
        store,
        key,
        std::sync::Arc::default(),
        Metrics::default(),
    )?;
    server::shutdown_signal().await?;
    info!("shutdown signal received; releasing in-flight jobs");
    module.shutdown().await;
    Ok(())
}

#[cfg(feature = "eval")]
fn run_backfill(args: &BackfillArgs) -> Result<()> {
    use inference_gateway::eval::backfill::backfill;
    use inference_gateway::store::{open_writer, unix_ms};

    let (store, eval) = eval_config(&args.config)?;
    let evaluator = eval
        .evaluators
        .iter()
        .find(|e| e.name == args.evaluator)
        .with_context(|| format!("no evaluator named {:?}", args.evaluator))?;
    let since = args.since.map_or(0, unix_ms);
    let mut conn = open_writer(store.path())?;
    let r = backfill(
        &mut conn,
        evaluator,
        since,
        args.all,
        args.dry_run,
        unix_ms(std::time::SystemTime::now()),
    )?;
    println!(
        "{} {}: {} matched, {} sampled, {} {}",
        evaluator.name,
        evaluator.version,
        r.matched,
        r.sampled,
        r.enqueued,
        if args.dry_run {
            "would be enqueued"
        } else {
            "enqueued"
        }
    );
    Ok(())
}
