//! Sets `cost_usd` from a pricing table (`config/pricing.toml`).

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, PoisonError};

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::warn;

use super::{Module, Phase};
use crate::config::CostFile;
use crate::observe::record::RequestRecord;
use crate::providers::Usage;

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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PricingFile {
    models: HashMap<String, Price>,
}

pub struct Cost {
    prices: HashMap<String, Price>,
    warned: Mutex<HashSet<String>>,
}

impl Cost {
    /// # Errors
    ///
    /// Unreadable or invalid pricing file.
    pub fn new(cfg: &CostFile) -> Result<Self> {
        let path = &cfg.pricing_file;
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading pricing file {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("pricing file {}", path.display()))
    }

    fn parse(text: &str) -> Result<Self> {
        let file: PricingFile = toml::from_str(text)?;
        Ok(Self {
            prices: file.models,
            warned: Mutex::default(),
        })
    }

    fn price(&self, model: &str) -> Option<&Price> {
        self.prices
            .get(model)
            .or_else(|| self.prices.get(strip_date(model)?))
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
        let Some(model) = rec
            .response_model
            .as_deref()
            .or(rec.requested_model.as_deref())
        else {
            return;
        };
        if let Some(p) = self.price(model) {
            rec.cost_usd = cost(p, &rec.usage);
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

    fn table() -> Cost {
        Cost::parse(PRICING).unwrap()
    }

    #[test]
    fn shipped_pricing_file_parses() {
        assert!(table().prices.len() > 20);
    }

    #[test]
    fn matches_exact_and_dated_ids_only() {
        let t = table();
        let haiku = t.price("claude-haiku-4-5").copied();
        assert!(haiku.is_some());
        assert_eq!(t.price("claude-haiku-4-5-20251001").copied(), haiku);
        assert_eq!(
            t.price("gpt-5-mini-2025-08-07").copied(),
            t.price("gpt-5-mini").copied()
        );
        assert_ne!(
            t.price("gpt-5-mini-2025-08-07").copied(),
            t.price("gpt-5").copied()
        );
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
        let p = table().price("gpt-5").copied().unwrap();
        assert_eq!(cost(&p, &Usage::default()), None);
    }
}
