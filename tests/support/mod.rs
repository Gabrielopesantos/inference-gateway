#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

pub mod mock_upstream;

use std::fmt::Write as _;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use clap::Parser;
use inference_gateway::config::Config;
use inference_gateway::config::cli::{Cli, Command};
use inference_gateway::gateway::{Gateway, Secrets};
use inference_gateway::modules::{self, BuildCtx, CompletedExchange, Dispatcher, Module, Phase};
use inference_gateway::observe::record::RequestRecord;
use inference_gateway::server::{self, Handles};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

pub struct TestGateway {
    pub addr: SocketAddr,
    pub admin_addr: SocketAddr,
    pub handles: Handles,
    pub task: JoinHandle<anyhow::Result<()>>,
    pub ctx: BuildCtx,
    dispatcher: Dispatcher,
    records: mpsc::UnboundedReceiver<CompletedExchange>,
}

impl TestGateway {
    pub fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }

    pub fn admin_url(&self, path: &str) -> String {
        format!("http://{}{path}", self.admin_addr)
    }

    /// Next finished record; panics after 2 s.
    pub async fn record(&mut self) -> RequestRecord {
        self.exchange().await.record
    }

    /// Next finished exchange, with captured content; panics after 2 s.
    pub async fn exchange(&mut self) -> CompletedExchange {
        tokio::time::timeout(Duration::from_secs(2), self.records.recv())
            .await
            .expect("no record within 2 s")
            .expect("record channel closed")
    }

    /// Asserts no further record arrives within `wait`.
    pub async fn assert_no_record(&mut self, wait: Duration) {
        if let Ok(Some(r)) = tokio::time::timeout(wait, self.records.recv()).await {
            panic!("unexpected extra record: {:?}", r.record);
        }
    }

    pub async fn shutdown(self, grace: Duration) -> anyhow::Result<()> {
        self.handles.graceful_shutdown(grace);
        let served = self.task.await?;
        self.dispatcher.shutdown(grace).await;
        served
    }

    pub async fn metrics(&self) -> String {
        client()
            .get(self.admin_url("/metrics"))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .text()
            .await
            .unwrap()
    }
}

/// Forwards every exchange to the test.
struct Collector(mpsc::UnboundedSender<CompletedExchange>);

#[async_trait]
impl Module for Collector {
    fn name(&self) -> &'static str {
        "test_collector"
    }

    fn phase(&self) -> Phase {
        Phase::Export
    }

    async fn export(&self, ex: Arc<CompletedExchange>) {
        let _ = self.0.send((*ex).clone());
    }
}

#[derive(Default)]
pub struct Options<'a> {
    pub flags: &'a [&'a str],
    pub secrets: Secrets,
    pub modules: Vec<Box<dyn Module>>,
    /// Export queue capacity; default [`Dispatcher::QUEUE_CAPACITY`].
    pub capacity: Option<usize>,
    /// Judge key, instead of the env var named in `[eval.judge]`.
    pub judge_key: Option<String>,
}

/// Start a gateway on ephemeral ports from config text and extra CLI flags.
pub async fn spawn_gateway(config: &str, flags: &[&str], token: Option<&str>) -> TestGateway {
    spawn_gateway_with(
        config,
        Options {
            flags,
            secrets: Secrets {
                gateway_token: token.map(str::to_owned),
                identity_secret: None,
            },
            ..Options::default()
        },
    )
    .await
}

/// Config modules run, then `opts.modules`, then a collector for [`TestGateway::record`].
/// Async so it runs inside a runtime: the dispatcher spawns workers.
#[allow(clippy::unused_async)]
pub async fn spawn_gateway_with(config: &str, opts: Options<'_>) -> TestGateway {
    server::install_crypto_provider();
    let mut cli_args = vec!["ig", "--config", "unused.toml"];
    cli_args.extend_from_slice(opts.flags);
    let Command::Serve(args) = Cli::try_parse_from(cli_args)
        .unwrap()
        .into_command()
        .unwrap()
    else {
        panic!("not serve")
    };
    let config = Config::parse(config).unwrap();
    let mut ctx = BuildCtx::default();
    #[cfg(feature = "eval")]
    {
        ctx.judge_key = opts.judge_key;
    }
    let (tx, records) = mpsc::unbounded_channel();
    let mut mods = modules::build(&config, &mut ctx).unwrap();
    mods.extend(opts.modules);
    mods.push(Box::new(Collector(tx)));
    let dispatcher = Dispatcher::start(
        mods,
        &ctx,
        opts.capacity.unwrap_or(Dispatcher::QUEUE_CAPACITY),
    );
    let gw = Gateway::build(config, &args, opts.secrets, &dispatcher).unwrap();

    let proxy = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let admin = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = proxy.local_addr().unwrap();
    let admin_addr = admin.local_addr().unwrap();
    let handles = Handles::default();
    let task = tokio::spawn(server::serve(gw, proxy, admin, None, handles.clone()));
    TestGateway {
        addr,
        admin_addr,
        handles,
        task,
        ctx,
        dispatcher,
        records,
    }
}

/// Config with one upstream per `(name, provider, base_url)`.
pub fn config(upstreams: &[(&str, &str, &str)], extra: &str) -> String {
    let mut out = String::new();
    for (name, provider, base_url) in upstreams {
        writeln!(
            out,
            "[[upstreams]]\nname = \"{name}\"\nprovider = \"{provider}\"\nbase_url = \"{base_url}\"\n"
        )
        .unwrap();
    }
    out.push_str(extra);
    out
}

pub fn client() -> reqwest::Client {
    server::install_crypto_provider();
    reqwest::Client::builder().build().unwrap()
}
