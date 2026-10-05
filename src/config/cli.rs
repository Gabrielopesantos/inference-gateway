//! Process settings: flags with `IG_*` env fallback. Secrets are env-only,
//! never flags, so they stay out of `ps`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

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
    /// Work the evaluation queue of a gateway's database.
    #[command(subcommand)]
    Eval(EvalCommand),
}

#[derive(Debug, Subcommand)]
pub enum EvalCommand {
    /// Run eval workers only, against the database in `[store]`.
    Run(EvalRunArgs),
    /// Enqueue jobs for already captured traces.
    Backfill(BackfillArgs),
}

#[derive(Debug, Clone, Args)]
pub struct EvalRunArgs {
    /// TOML file with `[store]` and `[eval]`.
    #[arg(long, env = "IG_CONFIG")]
    pub config: PathBuf,

    /// Default `tracing` filter directive; `RUST_LOG` overrides it.
    #[arg(long, env = "IG_LOG_LEVEL", default_value = "info")]
    pub log_level: String,

    /// Log line format.
    #[arg(long, env = "IG_LOG_FORMAT", value_enum, default_value_t = LogFormat::Text)]
    pub log_format: LogFormat,
}

#[derive(Debug, Clone, Args)]
pub struct BackfillArgs {
    /// TOML file with `[store]` and `[eval]`.
    #[arg(long, env = "IG_CONFIG")]
    pub config: PathBuf,

    /// Evaluator name from `[[eval.evaluators]]`.
    #[arg(long)]
    pub evaluator: String,

    /// Only traces started since then: a duration ago (`7d`) or an RFC 3339
    /// time (`2026-10-01`, `2026-10-01T12:00:00Z`).
    #[arg(long, value_parser = parse_since)]
    pub since: Option<SystemTime>,

    /// Ignore the evaluator's `sample` and enqueue every matching trace.
    #[arg(long)]
    pub all: bool,

    /// Count, enqueue nothing.
    #[arg(long)]
    pub dry_run: bool,
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

fn parse_since(s: &str) -> Result<SystemTime, String> {
    if let Ok(ago) = humantime::parse_duration(s) {
        return SystemTime::now()
            .checked_sub(ago)
            .ok_or_else(|| format!("{s:?} is before the epoch"));
    }
    let full = if s.len() == 10 {
        format!("{s}T00:00:00Z")
    } else {
        s.to_owned()
    };
    humantime::parse_rfc3339_weak(&full)
        .map_err(|_| format!("{s:?} is neither a duration (7d) nor an RFC 3339 time"))
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
        let Command::Serve(args) = cli.into_command().unwrap() else {
            panic!("not serve")
        };
        assert_eq!(args.max_body, ByteSize::mib(32));
        assert_eq!(args.idle_timeout, Duration::from_mins(5));
        assert_eq!(args.log_format, LogFormat::Text);
    }

    #[test]
    fn no_subcommand_means_serve() {
        let Command::Serve(args) = Cli::try_parse_from(["ig", "--config", "gw.toml"])
            .unwrap()
            .into_command()
            .unwrap()
        else {
            panic!("not serve")
        };
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
        let Command::Serve(args) = cli.into_command().unwrap() else {
            panic!("not serve")
        };
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

    #[test]
    fn eval_subcommands_parse() {
        let cli = Cli::try_parse_from(["ig", "eval", "run", "--config", "gw.toml"]).unwrap();
        assert!(matches!(
            cli.into_command().unwrap(),
            Command::Eval(EvalCommand::Run(_))
        ));
        let cli = Cli::try_parse_from([
            "ig",
            "eval",
            "backfill",
            "--config",
            "gw.toml",
            "--evaluator",
            "answered",
            "--since",
            "2026-10-01",
            "--dry-run",
        ])
        .unwrap();
        let Command::Eval(EvalCommand::Backfill(b)) = cli.into_command().unwrap() else {
            panic!("not backfill")
        };
        assert_eq!(b.evaluator, "answered");
        assert!(b.dry_run && !b.all);
        assert_eq!(
            b.since.unwrap(),
            humantime::parse_rfc3339("2026-10-01T00:00:00Z").unwrap()
        );
    }

    #[test]
    fn since_takes_a_duration_ago() {
        let t = parse_since("1h").unwrap();
        let ago = SystemTime::now().duration_since(t).unwrap();
        assert!((Duration::from_mins(59)..Duration::from_mins(61)).contains(&ago));
        assert!(parse_since("yesterday").is_err());
    }
}
