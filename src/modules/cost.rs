//! Sets `cost_usd` from a fetched public price list. Models the list lacks
//! can be priced in `[cost.models]`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde_json::Value;
use tokio::task::JoinHandle;
use tracing::{info, warn};

use super::{Module, Phase};
use crate::config::{CostFile, Price, Tier};
use crate::observe::record::RequestRecord;
use crate::providers::Usage;

const FETCH_TIMEOUT: Duration = Duration::from_secs(30);
/// The list is about 3 MiB.
const MAX_LIST_BYTES: usize = 16 * 1024 * 1024;
/// A list with fewer priced models is an error page or a truncated file.
const MIN_MODELS: usize = 10;
/// Wait after a failed fetch while no list has loaded.
const RETRY_WITHOUT_LIST: Duration = Duration::from_mins(1);

type Prices = HashMap<String, Price>;

/// Swapped whole on refresh; readers clone the `Arc`.
type Fetched = Arc<RwLock<Arc<Prices>>>;
type Warned = Arc<Mutex<HashSet<String>>>;

pub struct Cost {
    /// From `source_url`; empty until the cache file or first fetch loads.
    fetched: Fetched,
    /// From `[cost.models]`; used for models the list lacks.
    fallback: Prices,
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
    /// Starts the refresh task: call inside a tokio runtime. Neither the
    /// first fetch nor a failed one blocks or fails.
    ///
    /// # Errors
    ///
    /// HTTP client build failure.
    pub fn new(cfg: &CostFile) -> Result<Self> {
        let mut cost = Self::with(Prices::new(), cfg.models.clone());
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
            cfg.source_url.clone(),
            cfg.refresh,
            cfg.cache_file.clone(),
            cost.fetched.clone(),
            cost.warned.clone(),
        )));
        Ok(cost)
    }

    fn with(fetched: Prices, fallback: Prices) -> Self {
        Self {
            fetched: Arc::new(RwLock::new(Arc::new(fetched))),
            fallback,
            warned: Warned::default(),
            refresher: None,
        }
    }

    fn price(&self, model: &str) -> Option<Price> {
        let fetched = self
            .fetched
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        find(&fetched, model).or_else(|| find(&self.fallback, model))
    }
}

/// Exact id, else the id without its date suffix.
fn find(prices: &Prices, model: &str) -> Option<Price> {
    prices
        .get(model)
        .or_else(|| prices.get(strip_date(model)?))
        .copied()
}

/// Per token → per million, without float noise such as 0.24999999999999997.
fn per_mtok(entry: &Value, key: &str) -> Option<f64> {
    entry
        .get(key)
        .and_then(Value::as_f64)
        .map(|usd| (usd * 1e12).round() / 1e6)
}

/// `LiteLLM`'s `model_prices_and_context_window.json`: model id → object
/// with USD per token. Entries without token prices (images, audio) are
/// skipped; batch, flex and priority prices are not read.
fn parse_litellm(bytes: &[u8]) -> Result<Prices> {
    let list: HashMap<String, Value> = serde_json::from_slice(bytes)?;
    let prices: Prices = list
        .iter()
        // Documents the format.
        .filter(|(model, _)| model.as_str() != "sample_spec")
        .filter_map(|(model, e)| {
            let output = per_mtok(e, "output_cost_per_token")?;
            let price = Price {
                input: per_mtok(e, "input_cost_per_token")?,
                output,
                cache_read: per_mtok(e, "cache_read_input_token_cost"),
                cache_write: per_mtok(e, "cache_creation_input_token_cost"),
                cache_write_1h: per_mtok(e, "cache_creation_input_token_cost_above_1hr"),
                tier: tier(e, output),
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

/// Long-prompt prices: the base keys with an `_above_{N}k_tokens` suffix. An
/// entry has at most one threshold.
fn tier(entry: &Value, base_output: f64) -> Option<Tier> {
    let (suffix, above) = entry.as_object()?.keys().find_map(|key| {
        let suffix = key.strip_prefix("input_cost_per_token")?;
        let thousands = suffix.strip_prefix("_above_")?.strip_suffix("k_tokens")?;
        Some((suffix, thousands.parse::<u64>().ok()? * 1000))
    })?;
    let at = |key: &str| per_mtok(entry, &format!("{key}{suffix}"));
    Some(Tier {
        above,
        input: at("input_cost_per_token")?,
        output: at("output_cost_per_token").unwrap_or(base_output),
        cache_read: at("cache_read_input_token_cost"),
        cache_write: at("cache_creation_input_token_cost"),
        cache_write_1h: at("cache_creation_input_token_cost_above_1hr"),
    })
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

/// Time to the next fetch: sooner while there are no prices to fall back on.
fn next_fetch(every: Duration, has_list: bool) -> Duration {
    if has_list {
        every
    } else {
        every.min(RETRY_WITHOUT_LIST)
    }
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
    loop {
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
        let has_list = !fetched
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty();
        tokio::time::sleep(next_fetch(every, has_list)).await;
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

/// The tier's rates when the prompt exceeds its threshold; they apply to the
/// whole request.
fn rates(price: &Price, prompt_tokens: u64) -> Price {
    match price.tier {
        Some(t) if prompt_tokens > t.above => Price {
            input: t.input,
            output: t.output,
            cache_read: t.cache_read.or(price.cache_read),
            cache_write: t.cache_write.or(price.cache_write),
            cache_write_1h: t.cache_write_1h.or(price.cache_write_1h),
            // Already applied; must not tier again if this result is re-priced.
            tier: None,
        },
        _ => *price,
    }
}

#[allow(clippy::cast_precision_loss)]
fn cost(price: &Price, u: &Usage) -> Option<f64> {
    if u.input_tokens_total.is_none() && u.output_tokens.is_none() {
        return None;
    }
    let total = u.input_tokens_total.unwrap_or(0);
    let price = rates(price, total);
    let read = u.cache_read_tokens.unwrap_or(0);
    let write = u.cache_write_tokens.unwrap_or(0);
    let write_1h = u.cache_write_1h_tokens.unwrap_or(0).min(write);
    let uncached = total.saturating_sub(read + write);
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

    const LITELLM: &[u8] = include_bytes!("../../tests/fixtures/pricing/litellm.json");

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.unwrap();
        assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
    }

    fn flat(input: f64, output: f64) -> Price {
        Price {
            input,
            output,
            cache_read: None,
            cache_write: None,
            cache_write_1h: None,
            tier: None,
        }
    }

    fn table() -> Cost {
        Cost::with(parse_litellm(LITELLM).unwrap(), Prices::new())
    }

    #[test]
    fn litellm_list_maps_to_prices_per_million() {
        let prices = parse_litellm(LITELLM).unwrap();
        assert_eq!(
            prices["claude-sonnet-5"],
            Price {
                cache_read: Some(0.2),
                cache_write: Some(2.5),
                cache_write_1h: Some(4.0),
                ..flat(2.0, 10.0)
            }
        );
        assert_eq!(
            prices["gpt-5-mini"],
            Price {
                cache_read: Some(0.025),
                ..flat(0.25, 2.0)
            }
        );
        assert!(!prices.contains_key("sample_spec"));
    }

    #[test]
    fn litellm_list_maps_long_prompt_tiers() {
        let prices = parse_litellm(LITELLM).unwrap();
        assert_eq!(
            prices["gpt-5.5"].tier,
            Some(Tier {
                above: 272_000,
                input: 10.0,
                output: 45.0,
                cache_read: Some(1.0),
                cache_write: None,
                cache_write_1h: None,
            })
        );
        assert_eq!(
            prices["claude-sonnet-4-5"].tier,
            Some(Tier {
                above: 200_000,
                input: 6.0,
                output: 22.5,
                cache_read: Some(0.6),
                cache_write: Some(7.5),
                cache_write_1h: Some(12.0),
            })
        );
        assert_eq!(prices["gpt-5"].tier, None);
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
    fn list_wins_over_inline_prices() {
        let ours = flat(99.0, 99.0);
        let fallback = Prices::from([
            ("gpt-5-mini".to_owned(), ours),
            ("local-model".to_owned(), ours),
        ]);
        let t = Cost::with(parse_litellm(LITELLM).unwrap(), fallback);
        assert_eq!(t.price("gpt-5-mini").map(|p| p.input), Some(0.25));
        assert_eq!(
            t.price("gpt-5-mini-2025-08-07").map(|p| p.input),
            Some(0.25)
        );
        assert_eq!(t.price("local-model"), Some(ours));
        assert_eq!(t.price("local-model-2026-01-01"), Some(ours));
        assert!(t.price("gpt-unknown").is_none());
    }

    #[test]
    fn inline_prices_work_before_the_list_loads() {
        let ours = flat(1.0, 2.0);
        let t = Cost::with(Prices::new(), Prices::from([("gpt-5".to_owned(), ours)]));
        assert_eq!(t.price("gpt-5"), Some(ours));
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
    fn failed_fetch_is_retried_sooner_without_a_list() {
        let every = Duration::from_hours(6);
        assert_eq!(next_fetch(every, true), every);
        assert_eq!(next_fetch(every, false), RETRY_WITHOUT_LIST);
        let short = Duration::from_secs(5);
        assert_eq!(next_fetch(short, false), short);
    }

    #[test]
    fn matches_exact_and_dated_ids_only() {
        let t = table();
        let haiku = t.price("claude-haiku-4-5");
        assert!(haiku.is_some());
        assert_eq!(t.price("claude-haiku-4-5-20251001"), haiku);
        assert_eq!(t.price("gpt-5-nano-2025-08-07"), t.price("gpt-5-nano"));
        assert_ne!(t.price("gpt-5-nano-2025-08-07"), t.price("gpt-5"));
        assert!(t.price("gpt-5-turbo").is_none());
        assert!(t.price("gpt-5-1").is_none());
        assert!(t.price("Qwen/Qwen3-8B").is_none());
    }

    #[test]
    fn cache_tokens_use_cache_prices() {
        let p = Price {
            cache_read: Some(0.2),
            cache_write: Some(2.5),
            cache_write_1h: Some(4.0),
            ..flat(2.0, 10.0)
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
        let p = flat(1.0, 0.0);
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
    fn long_prompt_is_billed_at_tier_rates() {
        let p = table().price("gpt-5.5").unwrap();
        let usage = |prompt| Usage {
            input_tokens_total: Some(prompt),
            cache_read_tokens: Some(100_000),
            output_tokens: Some(1_000),
            ..Usage::default()
        };
        // At the threshold: $5 input, $0.50 cached, $30 output.
        assert_close(
            cost(&p, &usage(272_000)),
            (172_000.0 * 5.0 + 100_000.0 * 0.5 + 1_000.0 * 30.0) / 1e6,
        );
        // Above it, the whole request: $10 input, $1 cached, $45 output.
        assert_close(
            cost(&p, &usage(272_001)),
            (172_001.0 * 10.0 + 100_000.0 * 1.0 + 1_000.0 * 45.0) / 1e6,
        );
    }

    #[test]
    fn no_usage_means_no_cost() {
        let p = table().price("gpt-5").unwrap();
        assert_eq!(cost(&p, &Usage::default()), None);
    }
}
