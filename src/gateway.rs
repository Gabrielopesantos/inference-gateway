//! Immutable runtime state shared by every handler, built once from [`Config`].

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use ipnet::IpNet;

use crate::config::cli::{GATEWAY_TOKEN_ENV, ServeArgs};
use crate::config::{Config, Upstream};

#[derive(Debug)]
pub struct Gateway {
    upstreams: HashMap<String, Upstream>,
    pub client: reqwest::Client,
    pub max_body: u64,
    pub ingress: IngressPolicy,
}

#[derive(Debug, Default)]
pub struct IngressPolicy {
    pub allow_cidrs: Vec<IpNet>,
    /// `Some` when `x-gateway-token` is required.
    pub token: Option<String>,
}

impl Gateway {
    /// `token` comes from `IG_GATEWAY_TOKEN`, passed in so tests need not
    /// touch the process env.
    ///
    /// # Errors
    ///
    /// Token required but empty, or HTTP client build failure.
    pub fn build(config: Config, args: &ServeArgs, token: Option<String>) -> Result<Arc<Self>> {
        let token = if config.ingress.require_token {
            match token.filter(|t| !t.is_empty()) {
                Some(t) => Some(t),
                None => bail!("ingress.require_token is set but {GATEWAY_TOKEN_ENV} is empty"),
            }
        } else {
            None
        };

        // `read_timeout` resets on every read, so it bounds silence, not total
        // stream length. A 3xx goes back to the client as is.
        let client = reqwest::Client::builder()
            .connect_timeout(args.connect_timeout)
            .read_timeout(args.idle_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .context("building upstream HTTP client")?;

        let upstreams = config
            .upstreams
            .into_iter()
            .map(|u| (u.name.clone(), u))
            .collect();

        Ok(Arc::new(Self {
            upstreams,
            client,
            max_body: args.max_body.as_u64(),
            ingress: IngressPolicy {
                allow_cidrs: config.ingress.allow_cidrs,
                token,
            },
        }))
    }

    #[must_use]
    pub fn upstream(&self, name: &str) -> Option<&Upstream> {
        self.upstreams.get(name)
    }
}
