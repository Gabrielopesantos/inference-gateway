//! Immutable runtime state shared by every handler, built once from [`Config`].

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use anyhow::{Context, Result};
use ipnet::IpNet;

use crate::capture;
use crate::config::cli::{GATEWAY_TOKEN_ENV, ServeArgs};
use crate::config::{Config, Upstream};
use crate::identity::{IDENTITY_SECRET_ENV, IdentitySecret};
use crate::modules::{Dispatcher, Scrape};
use crate::observe::RecordSink;
use crate::providers::{self, ProviderAdapter};

pub struct Gateway {
    routes: HashMap<String, Route>,
    pub client: reqwest::Client,
    pub max_body: u64,
    pub ingress: IngressPolicy,
    pub identity: Option<IdentitySecret>,
    pub sink: RecordSink,
    pub metrics: Option<Scrape>,
    /// `None` without a `[capture]` section.
    pub capture: Option<capture::Policy>,
}

#[derive(Debug)]
pub struct Route {
    pub upstream: Upstream,
    pub adapter: Arc<dyn ProviderAdapter>,
}

impl fmt::Debug for Gateway {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gateway")
            .field("routes", &self.routes)
            .field("max_body", &self.max_body)
            .field("ingress", &self.ingress)
            .field("identity", &self.identity)
            .field("capture", &self.capture)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Default)]
pub struct IngressPolicy {
    pub allow_cidrs: Vec<IpNet>,
    /// `Some` when `x-gateway-token` is required.
    pub token: Option<String>,
}

/// Env-only settings, passed in so tests need not touch the process env.
#[derive(Debug, Default)]
pub struct Secrets {
    pub gateway_token: Option<String>,
    pub identity_secret: Option<String>,
}

impl Secrets {
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            gateway_token: std::env::var(GATEWAY_TOKEN_ENV).ok(),
            identity_secret: std::env::var(IDENTITY_SECRET_ENV).ok(),
        }
    }
}

impl Gateway {
    /// # Errors
    ///
    /// Token required but empty, or HTTP client build failure.
    pub fn build(
        config: Config,
        args: &ServeArgs,
        secrets: Secrets,
        dispatcher: &Dispatcher,
    ) -> Result<Arc<Self>> {
        let token = config
            .ingress
            .require_token
            .then(|| {
                secrets
                    .gateway_token
                    .filter(|t| !t.is_empty())
                    .with_context(|| {
                        format!("ingress.require_token is set but {GATEWAY_TOKEN_ENV} is empty")
                    })
            })
            .transpose()?;

        // `read_timeout` resets on every read, so it bounds silence, not total
        // stream length. A 3xx goes back to the client as is.
        let client = reqwest::Client::builder()
            .connect_timeout(args.connect_timeout)
            .read_timeout(args.idle_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .context("building upstream HTTP client")?;

        let routes = config
            .upstreams
            .into_iter()
            .map(|upstream| {
                let adapter = providers::adapter(&upstream);
                (upstream.name.clone(), Route { upstream, adapter })
            })
            .collect();

        Ok(Arc::new(Self {
            routes,
            client,
            max_body: args.max_body.as_u64(),
            ingress: IngressPolicy {
                allow_cidrs: config.ingress.allow_cidrs,
                token,
            },
            identity: secrets
                .identity_secret
                .as_deref()
                .and_then(IdentitySecret::new),
            sink: dispatcher.sink(),
            metrics: dispatcher.scrape(),
            capture: config.capture,
        }))
    }

    #[must_use]
    pub fn route(&self, name: &str) -> Option<&Route> {
        self.routes.get(name)
    }
}
