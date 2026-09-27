#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

pub mod mock_upstream;

use std::fmt::Write as _;
use std::net::SocketAddr;
use std::time::Duration;

use clap::Parser;
use inference_gateway::config::Config;
use inference_gateway::config::cli::{Cli, Command};
use inference_gateway::gateway::Gateway;
use inference_gateway::server::{self, Handles};
use tokio::task::JoinHandle;

pub struct TestGateway {
    pub addr: SocketAddr,
    pub admin_addr: SocketAddr,
    pub handles: Handles,
    pub task: JoinHandle<anyhow::Result<()>>,
}

impl TestGateway {
    pub fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }

    pub fn admin_url(&self, path: &str) -> String {
        format!("http://{}{path}", self.admin_addr)
    }

    pub async fn shutdown(self, grace: Duration) -> anyhow::Result<()> {
        self.handles.graceful_shutdown(grace);
        self.task.await?
    }
}

/// Start a gateway on ephemeral ports from config text and extra CLI flags.
pub async fn spawn_gateway(config: &str, flags: &[&str], token: Option<&str>) -> TestGateway {
    server::install_crypto_provider();
    let mut cli_args = vec!["ig", "--config", "unused.toml"];
    cli_args.extend_from_slice(flags);
    let Command::Serve(args) = Cli::try_parse_from(cli_args)
        .unwrap()
        .into_command()
        .unwrap();
    let gw = Gateway::build(
        Config::parse(config).unwrap(),
        &args,
        token.map(str::to_owned),
    )
    .unwrap();

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
