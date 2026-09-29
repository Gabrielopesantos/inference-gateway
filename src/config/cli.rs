//! Process settings: flags with `IG_*` env fallback. Secrets are env-only,
//! never flags, so they stay out of `ps`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use bytesize::ByteSize;
use clap::{Args, Parser, Subcommand, ValueEnum};

pub const GATEWAY_TOKEN_ENV: &str = "IG_GATEWAY_TOKEN";

#[derive(Debug, Parser)]
#[command(
    name = "inference-gateway",
    version,
    about,
    args_conflicts_with_subcommands = true
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Flags for `serve`, accepted without the subcommand too.
    #[command(flatten)]
    pub serve: Option<ServeArgs>,
}

impl Cli {
    /// No subcommand means `serve`.
    ///
    /// # Errors
    ///
    /// Neither a subcommand nor serve flags; clap normally rejects this first.
    pub fn into_command(self) -> Result<Command, clap::Error> {
        match (self.command, self.serve) {
            (Some(command), _) => Ok(command),
            (None, Some(serve)) => Ok(Command::Serve(serve)),
            (None, None) => Err(clap::Error::new(
                clap::error::ErrorKind::MissingRequiredArgument,
            )),
        }
    }
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the gateway (the default when no subcommand is given).
    Serve(ServeArgs),
}

#[derive(Debug, Clone, Args)]
pub struct ServeArgs {
    /// TOML file with upstreams and domain config.
    #[arg(long, env = "IG_CONFIG")]
    pub config: PathBuf,

    /// Proxy listen address.
    #[arg(long, env = "IG_LISTEN", default_value = "127.0.0.1:8080")]
    pub listen: SocketAddr,

    /// Admin listen address (`/healthz`, `/metrics`).
    #[arg(long, env = "IG_ADMIN_LISTEN", default_value = "127.0.0.1:9090")]
    pub admin_listen: SocketAddr,

    /// Max request body size; larger bodies get 413.
    #[arg(long, env = "IG_MAX_BODY", default_value = "32 MiB")]
    pub max_body: ByteSize,

    /// Timeout for opening a connection to an upstream.
    #[arg(long, env = "IG_CONNECT_TIMEOUT", default_value = "10s", value_parser = parse_duration)]
    pub connect_timeout: Duration,

    /// Max silence from an upstream, before the response headers and between
    /// body chunks.
    #[arg(long, env = "IG_IDLE_TIMEOUT", default_value = "300s", value_parser = parse_duration)]
    pub idle_timeout: Duration,

    /// On SIGTERM/SIGINT, how long in-flight streams may finish before the
    /// gateway cuts them.
    #[arg(long, env = "IG_SHUTDOWN_TIMEOUT", default_value = "30s", value_parser = parse_duration)]
    pub shutdown_timeout: Duration,

    /// PEM certificate chain; serves TLS when set together with `--tls-key`.
    #[arg(long, env = "IG_TLS_CERT", requires = "tls_key")]
    pub tls_cert: Option<PathBuf>,

    /// PEM private key for `--tls-cert`.
    #[arg(long, env = "IG_TLS_KEY", requires = "tls_cert")]
    pub tls_key: Option<PathBuf>,

    /// Default `tracing` filter directive; `RUST_LOG` overrides it.
    #[arg(long, env = "IG_LOG_LEVEL", default_value = "info")]
    pub log_level: String,

    /// Log line format.
    #[arg(long, env = "IG_LOG_FORMAT", value_enum, default_value_t = LogFormat::Text)]
    pub log_format: LogFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LogFormat {
    Text,
    Json,
}

fn parse_duration(s: &str) -> Result<Duration, humantime::DurationError> {
    humantime::parse_duration(s)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn defaults_parse() {
        let cli = Cli::try_parse_from(["ig", "--config", "gw.toml"]).unwrap();
        let Command::Serve(args) = cli.into_command().unwrap();
        assert_eq!(args.max_body, ByteSize::mib(32));
        assert_eq!(args.idle_timeout, Duration::from_mins(5));
        assert_eq!(args.log_format, LogFormat::Text);
    }

    #[test]
    fn no_subcommand_means_serve() {
        let Command::Serve(args) = Cli::try_parse_from(["ig", "--config", "gw.toml"])
            .unwrap()
            .into_command()
            .unwrap();
        assert_eq!(args.config, PathBuf::from("gw.toml"));
    }

    #[test]
    fn serve_subcommand_takes_the_same_flags() {
        let cli = Cli::try_parse_from([
            "ig",
            "serve",
            "--config",
            "gw.toml",
            "--connect-timeout",
            "2s",
        ])
        .unwrap();
        let Command::Serve(args) = cli.into_command().unwrap();
        assert_eq!(args.connect_timeout, Duration::from_secs(2));
    }

    #[test]
    fn config_is_required() {
        assert!(Cli::try_parse_from(["ig"]).is_err());
        assert!(Cli::try_parse_from(["ig", "serve"]).is_err());
    }

    #[test]
    fn tls_needs_both_cert_and_key() {
        let parse = |extra: &[&str]| {
            let mut argv = vec!["ig", "--config", "gw.toml"];
            argv.extend_from_slice(extra);
            Cli::try_parse_from(argv)
        };
        assert!(parse(&["--tls-cert", "c.pem"]).is_err());
        assert!(parse(&["--tls-key", "k.pem"]).is_err());
        assert!(parse(&["--tls-cert", "c.pem", "--tls-key", "k.pem"]).is_ok());
    }
}
