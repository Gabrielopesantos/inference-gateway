//! Sets `cost_usd` from a pricing table: hand-maintained prices
//! (`config/pricing.toml`), a fetched public price list, or both.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tracing::{info, warn};

use super::{Module, Phase};
use crate::config::CostFile;
use crate::observe::record::RequestRecord;
use crate::providers::Usage;

const FETCH_TIMEOUT: Duration = Duration::from_secs(30);
/// The list is about 3 MiB.
const MAX_LIST_BYTES: usize = 16 * 1024 * 1024;
/// A list with fewer priced models is an error page or a truncated file.
const MIN_MODELS: usize = 10;

/// USD per million tokens.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Price {
    pub input: f64,
    pub output: f64,
    pub cache_read: Option<f64>,
    pub cache_write: Option<f64>,
    pub cache_write_1h: Option<f64>,
}

type Prices = HashMap<String, Price>;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PricingFile {
    models: Prices,
}

/// Swapped whole on refresh; readers clone the `Arc`.
type Fetched = Arc<RwLock<Arc<Prices>>>;
type Warned = Arc<Mutex<HashSet<String>>>;

pub struct Cost {
    /// From `pricing_file`; win per model.
    overrides: Prices,
    /// From `source_url`; empty until the cache file or first fetch loads.
    fetched: Fetched,
    warned: Warned,
    refresher: Option<JoinHandle<()>>,
}

impl Drop for Cost {
    fn drop(&mut self) {
        if let Some(task) = &self.refresher {
            task.abort();
        }
    }
}

impl Cost {
    /// With `source_url`, starts the refresh task: call inside a tokio
    /// runtime. Neither the first fetch nor a failed one blocks or fails.
    ///
    /// # Errors
    ///
    /// Unreadable or invalid pricing file, or HTTP client build failure.
    pub fn new(cfg: &CostFile) -> Result<Self> {
        let overrides = match &cfg.pricing_file {
            Some(path) => {
                let text = std::fs::read_to_string(path)
                    .with_context(|| format!("reading pricing file {}", path.display()))?;
                parse_file(&text).with_context(|| format!("pricing file {}", path.display()))?
            }
            None => Prices::new(),
        };
        let mut cost = Self::with(overrides, Prices::new());
        let Some(url) = &cfg.source_url else {
            return Ok(cost);
        };
        if let Some(cache) = &cfg.cache_file {
            match load_cache(cache) {
                Ok(Some(prices)) => {
                    info!(models = prices.len(), cache = %cache.display(), "prices loaded from cache");
                    cost.fetched = Arc::new(RwLock::new(Arc::new(prices)));
                }
                Ok(None) => {}
                Err(e) => {
                    warn!(cache = %cache.display(), error = %format!("{e:#}"), "price cache ignored");
                }
            }
        }
        let client = reqwest::Client::builder()
            .timeout(FETCH_TIMEOUT)
            .build()
            .context("building price list HTTP client")?;
        cost.refresher = Some(tokio::spawn(refresh_loop(
            client,
            url.clone(),
            cfg.refresh,
            cfg.cache_file.clone(),
            cost.fetched.clone(),
            cost.warned.clone(),
        )));
        Ok(cost)
    }

    fn with(overrides: Prices, fetched: Prices) -> Self {
        Self {
            overrides,
            fetched: Arc::new(RwLock::new(Arc::new(fetched))),
            warned: Warned::default(),
            refresher: None,
        }
    }

    fn price(&self, model: &str) -> Option<Price> {
        find(&self.overrides, model).or_else(|| {
            let fetched = self
                .fetched
                .read()
                .unwrap_or_else(PoisonError::into_inner)
                .clone();
            find(&fetched, model)
        })
    }
}

/// Exact id, else the id without its date suffix.
fn find(prices: &Prices, model: &str) -> Option<Price> {
    prices
        .get(model)
        .or_else(|| prices.get(strip_date(model)?))
        .copied()
}

fn parse_file(text: &str) -> Result<Prices> {
    let file: PricingFile = toml::from_str(text)?;
    Ok(file.models)
}

/// `LiteLLM`'s `model_prices_and_context_window.json`: model id → object
/// with USD per token. Entries without token prices (images, audio) are
/// skipped; tiered prices (`*_above_200k_tokens`, batch, priority) are not read.
fn parse_litellm(bytes: &[u8]) -> Result<Prices> {
    let list: HashMap<String, Value> = serde_json::from_slice(bytes)?;
    // Per token → per million, without float noise such as 0.24999999999999997.
    let per_mtok = |entry: &Value, key: &str| {
        entry
            .get(key)
            .and_then(Value::as_f64)
            .map(|usd| (usd * 1e12).round() / 1e6)
    };
    let prices: Prices = list
        .iter()
        // Documents the format.
        .filter(|(model, _)| model.as_str() != "sample_spec")
        .filter_map(|(model, e)| {
            let price = Price {
                input: per_mtok(e, "input_cost_per_token")?,
                output: per_mtok(e, "output_cost_per_token")?,
                cache_read: per_mtok(e, "cache_read_input_token_cost"),
                cache_write: per_mtok(e, "cache_creation_input_token_cost"),
                cache_write_1h: per_mtok(e, "cache_creation_input_token_cost_above_1hr"),
            };
            Some((model.clone(), price))
        })
        .collect();
    if prices.len() < MIN_MODELS {
        bail!(
            "only {} priced models; expected at least {MIN_MODELS}",
            prices.len()
        );
    }
    Ok(prices)
}

/// `None` when there is no cache yet.
fn load_cache(path: &Path) -> Result<Option<Prices>> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(parse_litellm(&bytes)?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Temp file and rename, so a crash never leaves half a cache.
fn write_cache(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

async fn fetch(client: &reqwest::Client, url: &str) -> Result<Vec<u8>> {
    let mut resp = client.get(url).send().await?.error_for_status()?;
    let mut body = Vec::new();
    while let Some(chunk) = resp.chunk().await? {
        if body.len() + chunk.len() > MAX_LIST_BYTES {
            bail!("price list larger than {MAX_LIST_BYTES} bytes");
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Fetches now, then every `every`. A failed fetch keeps the current prices.
async fn refresh_loop(
    client: reqwest::Client,
    url: String,
    every: Duration,
    cache: Option<PathBuf>,
    fetched: Fetched,
    warned: Warned,
) {
    let mut ticks = tokio::time::interval(every);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        ticks.tick().await;
        let result = async {
            let bytes = fetch(&client, &url).await?;
            Ok::<_, anyhow::Error>((parse_litellm(&bytes)?, bytes))
        }
        .await;
        match result {
            Ok((prices, bytes)) => {
                info!(models = prices.len(), url, "prices refreshed");
                *fetched.write().unwrap_or_else(PoisonError::into_inner) = Arc::new(prices);
                // Models priced now go quiet; ones still missing are reported once more.
                warned
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .clear();
                if let Some(path) = cache.clone() {
                    let written =
                        tokio::task::spawn_blocking(move || write_cache(&path, &bytes)).await;
                    if let Ok(Err(e)) = written {
                        warn!(error = %format!("{e:#}"), "price cache not written");
                    }
                }
            }
            Err(e) => {
                warn!(url, error = %format!("{e:#}"), "price refresh failed; keeping current prices");
            }
        }
    }
}

/// `claude-haiku-4-5-20251001` -> `claude-haiku-4-5`, `gpt-4o-2024-08-06` -> `gpt-4o`.
fn strip_date(model: &str) -> Option<&str> {
    model
        .match_indices('-')
        .find(|(i, _)| is_date(&model[i + 1..]))
        .map(|(i, _)| &model[..i])
}

/// `YYYYMMDD` or `YYYY-MM-DD`.
fn is_date(s: &str) -> bool {
    let b = s.as_bytes();
    match b.len() {
        8 => b.iter().all(u8::is_ascii_digit),
        10 => b.iter().enumerate().all(|(i, c)| {
            if i == 4 || i == 7 {
                *c == b'-'
            } else {
                c.is_ascii_digit()
            }
        }),
        _ => false,
    }
}

#[allow(clippy::cast_precision_loss)]
fn cost(price: &Price, u: &Usage) -> Option<f64> {
    if u.input_tokens_total.is_none() && u.output_tokens.is_none() {
        return None;
    }
    let read = u.cache_read_tokens.unwrap_or(0);
    let write = u.cache_write_tokens.unwrap_or(0);
    let write_1h = u.cache_write_1h_tokens.unwrap_or(0).min(write);
    let uncached = u
        .input_tokens_total
        .unwrap_or(0)
        .saturating_sub(read + write);
    let output = u.output_tokens.unwrap_or(0);
    let write_5m_price = price.cache_write.unwrap_or(price.input);
    let usd = uncached as f64 * price.input
        + read as f64 * price.cache_read.unwrap_or(price.input)
        + (write - write_1h) as f64 * write_5m_price
        + write_1h as f64 * price.cache_write_1h.unwrap_or(write_5m_price)
        + output as f64 * price.output;
    Some(usd / 1e6)
}

impl Module for Cost {
    fn name(&self) -> &'static str {
        "cost"
    }

    fn phase(&self) -> Phase {
        Phase::Enrich
    }

    fn enrich(&self, rec: &mut RequestRecord) {
        let Some(model) = rec.model() else {
            return;
        };
        if let Some(p) = self.price(model) {
            rec.cost_usd = cost(&p, &rec.usage);
            return;
        }
        let mut warned = self.warned.lock().unwrap_or_else(PoisonError::into_inner);
        if warned.insert(model.to_owned()) {
            warn!(model, "no price for model; cost_usd stays empty");
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const PRICING: &str = include_str!("../../config/pricing.toml");

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.unwrap();
        assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
    }

    const LITELLM: &[u8] = include_bytes!("../../tests/fixtures/pricing/litellm.json");

    fn table() -> Cost {
        Cost::with(parse_file(PRICING).unwrap(), Prices::new())
    }

    #[test]
    fn shipped_pricing_file_parses() {
        assert!(table().overrides.len() > 20);
    }

    #[test]
    fn litellm_list_maps_to_prices_per_million() {
        let prices = parse_litellm(LITELLM).unwrap();
        assert_eq!(
            prices["claude-sonnet-5"],
            Price {
                input: 2.0,
                output: 10.0,
                cache_read: Some(0.2),
                cache_write: Some(2.5),
                cache_write_1h: Some(4.0),
            }
        );
        assert_eq!(
            prices["gpt-5-mini"],
            Price {
                input: 0.25,
                output: 2.0,
                cache_read: Some(0.025),
                cache_write: None,
                cache_write_1h: None,
            }
        );
        assert!(!prices.contains_key("sample_spec"));
    }

    /// The shipped file and the list agree where both have the model.
    #[test]
    fn litellm_list_agrees_with_shipped_file() {
        let file = parse_file(PRICING).unwrap();
        let list = parse_litellm(LITELLM).unwrap();
        let mut shared = 0;
        for (model, price) in &file {
            if let Some(listed) = list.get(model) {
                shared += 1;
                assert_eq!(
                    (listed.input, listed.output, listed.cache_read),
                    (price.input, price.output, price.cache_read),
                    "{model}"
                );
            }
        }
        assert!(shared >= 8, "{shared}");
    }

    #[test]
    fn short_or_broken_list_is_rejected() {
        let e = parse_litellm(
            br#"{"gpt-5": {"input_cost_per_token": 1e-6, "output_cost_per_token": 1e-6}}"#,
        )
        .unwrap_err();
        assert!(e.to_string().contains("only 1 priced models"), "{e}");
        assert!(parse_litellm(b"<html>not found</html>").is_err());
    }

    #[test]
    fn pricing_file_wins_over_the_list() {
        let ours = Price {
            input: 99.0,
            output: 99.0,
            cache_read: None,
            cache_write: None,
            cache_write_1h: None,
        };
        let overrides = Prices::from([("gpt-5-mini".to_owned(), ours)]);
        let t = Cost::with(overrides, parse_litellm(LITELLM).unwrap());
        assert_eq!(t.price("gpt-5-mini"), Some(ours));
        // The dated id is in the list, but the override covers it first.
        assert_eq!(t.price("gpt-5-mini-2025-08-07"), Some(ours));
        assert_eq!(t.price("gpt-5-nano").map(|p| p.input), Some(0.05));
        assert_eq!(t.price("o3-2025-04-16").map(|p| p.input), Some(2.0));
        assert!(t.price("gpt-unknown").is_none());
    }

    #[test]
    fn cache_file_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pricing-cache.json");
        assert!(load_cache(&path).unwrap().is_none());
        write_cache(&path, LITELLM).unwrap();
        assert_eq!(
            load_cache(&path).unwrap().unwrap(),
            parse_litellm(LITELLM).unwrap()
        );
        std::fs::write(&path, b"{").unwrap();
        assert!(load_cache(&path).is_err());
    }

    #[test]
    fn matches_exact_and_dated_ids_only() {
        let t = table();
        let haiku = t.price("claude-haiku-4-5");
        assert!(haiku.is_some());
        assert_eq!(t.price("claude-haiku-4-5-20251001"), haiku);
        assert_eq!(t.price("gpt-5-mini-2025-08-07"), t.price("gpt-5-mini"));
        assert_ne!(t.price("gpt-5-mini-2025-08-07"), t.price("gpt-5"));
        assert!(t.price("gpt-5-turbo").is_none());
        assert!(t.price("gpt-5-1").is_none());
        assert!(t.price("Qwen/Qwen3-8B").is_none());
    }

    #[test]
    fn cache_tokens_use_cache_prices() {
        let p = Price {
            input: 2.0,
            output: 10.0,
            cache_read: Some(0.2),
            cache_write: Some(2.5),
            cache_write_1h: Some(4.0),
        };
        let u = Usage {
            input_tokens_total: Some(1_000_000 + 2_000_000 + 4_000_000),
            output_tokens: Some(1_000_000),
            cache_read_tokens: Some(2_000_000),
            cache_write_tokens: Some(4_000_000),
            cache_write_1h_tokens: Some(3_000_000),
            reasoning_tokens: Some(500_000),
        };
        // 1M 5m writes at 2.5, 3M 1h writes at 4.
        assert_close(cost(&p, &u), 2.0 + 0.4 + 2.5 + 12.0 + 10.0);
    }

    #[test]
    fn missing_cache_prices_default_to_input() {
        let p = Price {
            input: 1.0,
            output: 0.0,
            cache_read: None,
            cache_write: None,
            cache_write_1h: None,
        };
        let u = Usage {
            input_tokens_total: Some(3_000_000),
            cache_read_tokens: Some(1_000_000),
            cache_write_tokens: Some(1_000_000),
            cache_write_1h_tokens: Some(500_000),
            ..Usage::default()
        };
        assert_close(cost(&p, &u), 3.0);
    }

    #[test]
    fn no_usage_means_no_cost() {
        let p = table().price("gpt-5").unwrap();
        assert_eq!(cost(&p, &Usage::default()), None);
    }
}
