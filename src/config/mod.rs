//! Domain config: TOML parsed into [`FileConfig`], then validated into
//! [`Config`], the only form code past startup sees.

pub mod cli;
pub mod eval;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use bytesize::ByteSize;
use globset::Glob;
use ipnet::IpNet;
use serde::Deserialize;
use url::Url;

use crate::capture;
use crate::providers::ProviderKind;

/// Unknown keys are errors, so a typo never silently disables a setting.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    pub upstreams: Vec<UpstreamFile>,
    #[serde(default)]
    pub ingress: IngressFile,
    pub cost: Option<CostFile>,
    pub prometheus: Option<PrometheusFile>,
    pub store: Option<StoreFile>,
    pub capture: Option<CaptureFile>,
    pub eval: Option<eval::EvalFile>,
    pub otel: Option<OtelFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamFile {
    pub name: String,
    pub provider: ProviderKind,
    pub base_url: String,
    pub inject_stream_usage: Option<bool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IngressFile {
    #[serde(default)]
    pub allow_cidrs: Vec<IpNet>,
    #[serde(default)]
    pub require_token: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostFile {
    /// Price list in `LiteLLM`'s `model_prices_and_context_window.json` format,
    /// fetched at startup and every `refresh`.
    #[serde(default = "default_price_source")]
    pub source_url: String,
    #[serde(default = "default_price_refresh", with = "humantime_serde")]
    pub refresh: Duration,
    /// Last good download, used until the first fetch succeeds. Defaults to
    /// `pricing-cache.json` next to the config file.
    pub cache_file: Option<PathBuf>,
    /// For models the list lacks; the list wins.
    #[serde(default)]
    pub models: HashMap<String, Price>,
}

/// USD per million tokens.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Price {
    pub input: f64,
    pub output: f64,
    pub cache_read: Option<f64>,
    pub cache_write: Option<f64>,
    pub cache_write_1h: Option<f64>,
    /// From the fetched list only.
    #[serde(skip)]
    pub tier: Option<Tier>,
}

/// Rates for a request whose prompt exceeds `above` tokens; absent cache
/// prices fall back to the base ones.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tier {
    pub above: u64,
    pub input: f64,
    pub output: f64,
    pub cache_read: Option<f64>,
    pub cache_write: Option<f64>,
    pub cache_write_1h: Option<f64>,
}

pub const DEFAULT_PRICE_SOURCE: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

fn default_price_source() -> String {
    DEFAULT_PRICE_SOURCE.into()
}

fn default_price_refresh() -> Duration {
    Duration::from_hours(6)
}

const DEFAULT_PRICE_CACHE: &str = "pricing-cache.json";

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrometheusFile {}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoreFile {
    pub path: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OtelFile {
    /// OTLP/HTTP base URL; spans go to `{endpoint}/v1/traces`.
    pub endpoint: String,
    #[serde(default = "default_service_name")]
    pub service_name: String,
    /// Share of traces exported when the client sent no `traceparent`.
    #[serde(default = "default_sample")]
    pub sample_ratio: f64,
    #[serde(default = "default_otel_flush", with = "humantime_serde")]
    pub flush: Duration,
    #[serde(default = "default_otel_timeout", with = "humantime_serde")]
    pub timeout: Duration,
}

fn default_service_name() -> String {
    "inference-gateway".into()
}

fn default_otel_flush() -> Duration {
    Duration::from_secs(2)
}

fn default_otel_timeout() -> Duration {
    Duration::from_secs(10)
}

#[derive(Debug, Clone)]
pub struct OtelConfig {
    pub traces_url: Url,
    pub service_name: String,
    pub sample_ratio: f64,
    pub flush: Duration,
    pub timeout: Duration,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureFile {
    #[serde(default = "default_max_bytes")]
    pub max_bytes: ByteSize,
    #[serde(default = "default_retention", with = "humantime_serde")]
    pub retention: Duration,
    #[serde(default)]
    pub rules: Vec<RuleFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuleFile {
    #[serde(rename = "match", default)]
    pub matcher: MatchFile,
    #[serde(default = "default_true")]
    pub capture: bool,
    #[serde(default = "default_sample")]
    pub sample: f64,
    #[serde(default)]
    pub allow_header_override: bool,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatchFile {
    pub upstream: Option<String>,
    pub model: Option<String>,
    /// Evaluators only, e.g. `["end_turn"]`.
    #[serde(default)]
    pub stop_reason: Vec<String>,
    /// Evaluators only: `false` skips requests for JSON output.
    pub structured_output: Option<bool>,
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
}

fn default_max_bytes() -> ByteSize {
    ByteSize::mib(8)
}

/// Content retention without a `[capture]` section, for content captured
/// by an earlier config.
pub const DEFAULT_RETENTION: Duration = Duration::from_hours(30 * 24);

fn default_retention() -> Duration {
    DEFAULT_RETENTION
}

fn default_true() -> bool {
    true
}

pub(crate) fn default_sample() -> f64 {
    1.0
}

/// Module sections stay ungated here, so a section for a disabled feature
/// still parses and `modules::build` can name the missing feature.
#[derive(Debug, Clone)]
pub struct Config {
    pub upstreams: Vec<Upstream>,
    pub ingress: IngressFile,
    pub cost: Option<CostFile>,
    pub prometheus: Option<PrometheusFile>,
    pub store: Option<StoreFile>,
    pub capture: Option<capture::Policy>,
    pub eval: Option<eval::EvalConfig>,
    pub otel: Option<OtelConfig>,
}

#[derive(Debug, Clone)]
pub struct Upstream {
    pub name: String,
    pub provider: ProviderKind,
    pub base_url: Url,
    pub inject_stream_usage: bool,
}

impl Config {
    /// # Errors
    ///
    /// Unreadable file, TOML syntax or type error, or failed validation.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let mut config =
            Self::parse(&text).with_context(|| format!("config {}", path.display()))?;
        // Relative paths in the file are relative to the file.
        if let Some(dir) = path.parent() {
            if let Some(cost) = &mut config.cost {
                let cache = cost.cache_file.take();
                cost.cache_file =
                    Some(dir.join(cache.unwrap_or_else(|| DEFAULT_PRICE_CACHE.into())));
            }
            if let Some(store) = &mut config.store {
                store.path = dir.join(&store.path);
            }
        }
        Ok(config)
    }

    /// # Errors
    ///
    /// See [`Config::load`].
    pub fn parse(text: &str) -> Result<Self> {
        let file: FileConfig = toml::from_str(text)?;
        Self::try_from(file)
    }
}

impl TryFrom<FileConfig> for Config {
    type Error = anyhow::Error;

    fn try_from(file: FileConfig) -> Result<Self> {
        if file.upstreams.is_empty() {
            bail!("at least one [[upstreams]] entry is required");
        }
        let mut seen = HashSet::new();
        let upstreams = file
            .upstreams
            .into_iter()
            .map(|u| {
                let name = u.name;
                validate_name(&name).with_context(|| format!("upstream {name:?}: name"))?;
                if !seen.insert(name.clone()) {
                    bail!("upstream {name:?}: duplicate name");
                }
                let base_url = parse_base_url(&u.base_url)
                    .with_context(|| format!("upstream {name:?}: base_url"))?;
                if u.inject_stream_usage.is_some() && u.provider != ProviderKind::Openai {
                    bail!("upstream {name:?}: inject_stream_usage applies only to provider openai");
                }
                Ok(Upstream {
                    name,
                    provider: u.provider,
                    base_url,
                    inject_stream_usage: u.inject_stream_usage.unwrap_or(true),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        if let Some(cost) = &file.cost {
            validate_cost(cost)?;
        }

        if file.capture.is_some() && file.store.is_none() {
            bail!("[capture] needs a [store] section");
        }
        let capture = file
            .capture
            .map(|c| capture_policy(c, &upstreams))
            .transpose()?;

        if file.eval.is_some() && file.store.is_none() {
            bail!("[eval] needs a [store] section");
        }
        let eval = file
            .eval
            .map(|e| eval::eval_config(e, &upstreams))
            .transpose()?;
        let otel = file.otel.map(otel_config).transpose()?;

        Ok(Self {
            upstreams,
            ingress: file.ingress,
            cost: file.cost,
            prometheus: file.prometheus,
            store: file.store,
            capture,
            eval,
            otel,
        })
    }
}

fn validate_cost(cost: &CostFile) -> Result<()> {
    let url = Url::parse(&cost.source_url).context("cost.source_url")?;
    if !matches!(url.scheme(), "http" | "https") {
        bail!(
            "cost.source_url: scheme must be http or https, got {:?}",
            url.scheme()
        );
    }
    if cost.refresh.is_zero() {
        bail!("cost.refresh must not be zero");
    }
    for (model, p) in &cost.models {
        let rates = [
            Some(p.input),
            Some(p.output),
            p.cache_read,
            p.cache_write,
            p.cache_write_1h,
        ];
        if !rates.into_iter().flatten().all(valid_price) {
            bail!("cost.models.{model:?}: prices must be finite and not negative");
        }
    }
    Ok(())
}

#[must_use]
pub fn valid_price(usd: f64) -> bool {
    usd.is_finite() && usd >= 0.0
}

fn otel_config(file: OtelFile) -> Result<OtelConfig> {
    let mut endpoint = parse_base_url(&file.endpoint).context("otel.endpoint")?;
    // `join` replaces the last segment unless the path ends in '/'.
    if !endpoint.path().ends_with('/') {
        let path = format!("{}/", endpoint.path());
        endpoint.set_path(&path);
    }
    let traces_url = endpoint.join("v1/traces").context("otel.endpoint")?;
    if !(0.0..=1.0).contains(&file.sample_ratio) {
        bail!(
            "otel.sample_ratio must be in [0, 1], got {}",
            file.sample_ratio
        );
    }
    if file.flush.is_zero() {
        bail!("otel.flush must not be zero");
    }
    if file.timeout.is_zero() {
        bail!("otel.timeout must not be zero");
    }
    if file.service_name.is_empty() {
        bail!("otel.service_name must not be empty");
    }
    Ok(OtelConfig {
        traces_url,
        service_name: file.service_name,
        sample_ratio: file.sample_ratio,
        flush: file.flush,
        timeout: file.timeout,
    })
}

fn capture_policy(file: CaptureFile, upstreams: &[Upstream]) -> Result<capture::Policy> {
    let rules = file
        .rules
        .into_iter()
        .enumerate()
        .map(|(i, r)| capture_rule(r, upstreams).with_context(|| format!("capture.rules[{i}]")))
        .collect::<Result<Vec<_>>>()?;
    Ok(capture::Policy {
        max_bytes: usize::try_from(file.max_bytes.as_u64()).unwrap_or(usize::MAX),
        retention: file.retention,
        rules,
    })
}

fn capture_rule(r: RuleFile, upstreams: &[Upstream]) -> Result<capture::Rule> {
    if !r.matcher.stop_reason.is_empty() {
        bail!("match.stop_reason is not known when capture is decided");
    }
    if r.matcher.structured_output.is_some() {
        bail!("match.structured_output is for evaluators only");
    }
    let matcher = parse_match(r.matcher, upstreams)?;
    if !(0.0..=1.0).contains(&r.sample) {
        bail!("sample must be in [0, 1], got {}", r.sample);
    }
    Ok(capture::Rule {
        matcher,
        capture: r.capture,
        sample: r.sample,
        allow_header_override: r.allow_header_override,
    })
}

fn parse_match(m: MatchFile, upstreams: &[Upstream]) -> Result<capture::Match> {
    if let Some(u) = &m.upstream
        && !upstreams.iter().any(|x| &x.name == u)
    {
        bail!("match.upstream: no upstream named {u:?}");
    }
    let model = m
        .model
        .map(|g| Glob::new(&g).map(|g| g.compile_matcher()))
        .transpose()
        .context("match.model")?;
    Ok(capture::Match {
        upstream: m.upstream,
        model,
        // Tag keys come from header names, which are lowercase.
        tags: m
            .tags
            .into_iter()
            .map(|(k, v)| (k.to_ascii_lowercase(), v))
            .collect(),
        stop_reasons: m.stop_reason,
        structured_output: m.structured_output,
    })
}

/// An upstream name is one URL path segment.
fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() || name == "." || name == ".." {
        bail!("must not be empty, \".\" or \"..\"");
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        bail!("only ASCII letters, digits, '-', '_' and '.' are allowed");
    }
    Ok(())
}

fn parse_base_url(raw: &str) -> Result<Url> {
    let url = Url::parse(raw)?;
    if !matches!(url.scheme(), "http" | "https") {
        bail!("scheme must be http or https, got {:?}", url.scheme());
    }
    if url.query().is_some() || url.fragment().is_some() {
        bail!("must not have a query or fragment");
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("must not carry credentials");
    }
    Ok(url)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const EXAMPLE: &str = include_str!("../../config/gateway.example.toml");

    fn err(text: &str) -> String {
        format!("{:#}", Config::parse(text).unwrap_err())
    }

    #[test]
    fn example_config_is_valid() {
        let cfg = Config::parse(EXAMPLE).unwrap();
        let names: Vec<_> = cfg.upstreams.iter().map(|u| u.name.as_str()).collect();
        assert_eq!(names, ["anthropic", "openai"]);
        assert_eq!(cfg.upstreams[0].provider, ProviderKind::Anthropic);
        assert_eq!(cfg.upstreams[1].provider, ProviderKind::Openai);
    }

    /// The commented-out eval sections, uncommented.
    fn with_eval(text: &str) -> String {
        let (head, eval) = text.split_at(text.find("# [eval.").unwrap());
        let mut out = head.to_owned();
        for l in eval.lines() {
            out.push_str(l.strip_prefix("# ").unwrap_or(l.trim_start_matches('#')));
            out.push('\n');
        }
        out
    }

    #[test]
    fn example_eval_sections_are_valid() {
        let deploy = include_str!("../../deploy/gateway.toml");
        for text in [EXAMPLE, deploy] {
            let eval = Config::parse(&with_eval(text)).unwrap().eval.unwrap();
            assert!(!eval.evaluators.is_empty());
        }
    }

    #[test]
    fn ingress_defaults_when_absent() {
        let cfg = Config::parse(
            r#"
            [[upstreams]]
            name = "a"
            provider = "anthropic"
            base_url = "https://api.anthropic.com"
            "#,
        )
        .unwrap();
        assert!(cfg.ingress.allow_cidrs.is_empty());
        assert!(!cfg.ingress.require_token);
    }

    #[test]
    fn unknown_key_reports_line_and_column() {
        let e = err(r#"
[[upstreams]]
name = "a"
provider = "anthropic"
base_url = "https://api.anthropic.com"
base_ulr = "typo"
"#);
        assert!(e.contains("line 6"), "{e}");
        assert!(e.contains("column 1"), "{e}");
        assert!(e.contains("base_ulr"), "{e}");
    }

    #[test]
    fn wrong_type_reports_line_and_column() {
        let e = err(r#"
[[upstreams]]
name = "a"
provider = "anthropic"
base_url = "https://api.anthropic.com"

[ingress]
require_token = "yes"
"#);
        assert!(e.contains("line 8"), "{e}");
        assert!(e.contains("column 17"), "{e}");
    }

    #[test]
    fn unknown_provider_is_rejected() {
        let e = err(r#"
[[upstreams]]
name = "a"
provider = "gemini"
base_url = "https://example.com"
"#);
        assert!(e.contains("line 4"), "{e}");
        assert!(e.contains("gemini"), "{e}");
    }

    #[test]
    fn duplicate_upstream_name_names_the_upstream() {
        let e = err(r#"
[[upstreams]]
name = "dup"
provider = "anthropic"
base_url = "https://api.anthropic.com"

[[upstreams]]
name = "dup"
provider = "openai"
base_url = "https://api.openai.com"
"#);
        assert!(e.contains(r#"upstream "dup": duplicate name"#), "{e}");
    }

    #[test]
    fn bad_url_names_the_upstream() {
        for bad in [
            "not a url",
            "ftp://example.com",
            "https://example.com/?a=1",
            "https://u:p@example.com",
        ] {
            let e = err(&format!(
                "[[upstreams]]\nname = \"broken\"\nprovider = \"openai\"\nbase_url = \"{bad}\"\n"
            ));
            assert!(e.contains(r#"upstream "broken": base_url"#), "{bad}: {e}");
        }
    }

    #[test]
    fn name_must_be_one_path_segment() {
        for bad in ["", "a/b", "a b"] {
            let e = err(&format!(
                "[[upstreams]]\nname = \"{bad}\"\nprovider = \"openai\"\nbase_url = \"https://x\"\n"
            ));
            assert!(e.contains("name"), "{bad:?}: {e}");
        }
    }

    #[test]
    fn inject_stream_usage_defaults_on_and_is_openai_only() {
        let cfg = Config::parse(EXAMPLE).unwrap();
        assert!(cfg.upstreams[1].inject_stream_usage);
        let e = err(r#"
[[upstreams]]
name = "a"
provider = "anthropic"
base_url = "https://x"
inject_stream_usage = false
"#);
        assert!(e.contains("only to provider openai"), "{e}");
    }

    #[test]
    fn module_sections_are_optional() {
        let cfg = Config::parse(EXAMPLE).unwrap();
        assert!(cfg.cost.is_some());
        assert!(cfg.prometheus.is_some());
        let cfg = Config::parse(
            "[[upstreams]]\nname = \"a\"\nprovider = \"openai\"\nbase_url = \"https://x\"\n",
        )
        .unwrap();
        assert!(cfg.cost.is_none());
        assert!(cfg.prometheus.is_none());
    }

    #[test]
    fn bare_cost_section_uses_the_public_list_and_a_cache_next_to_the_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gw.toml");
        let write = |extra: &str| {
            std::fs::write(&path, format!("{ONE}[cost]\n{extra}")).unwrap();
            Config::load(&path).unwrap().cost.unwrap()
        };
        let cost = write("");
        assert_eq!(cost.source_url, DEFAULT_PRICE_SOURCE);
        assert_eq!(cost.refresh, Duration::from_hours(6));
        assert_eq!(cost.cache_file, Some(dir.path().join("pricing-cache.json")));
        assert!(cost.models.is_empty());
        let cost = write("refresh = \"30m\"\ncache_file = \"p.json\"\n");
        assert_eq!(cost.refresh, Duration::from_mins(30));
        assert_eq!(cost.cache_file, Some(dir.path().join("p.json")));
    }

    #[test]
    fn cost_models_parse_and_reject_unknown_keys() {
        let model = "[cost.models.\"local\"]\ninput = 0.5\noutput = 1.5\n";
        let cost = Config::parse(&format!("{ONE}{model}"))
            .unwrap()
            .cost
            .unwrap();
        assert_eq!(
            cost.models["local"],
            Price {
                input: 0.5,
                output: 1.5,
                cache_read: None,
                cache_write: None,
                cache_write_1h: None,
                tier: None,
            }
        );
        let e = err(&format!("{ONE}{model}tier = 1\n"));
        assert!(e.contains("tier"), "{e}");
    }

    #[test]
    fn cost_source_url_must_be_http() {
        let e = err(&format!(
            "{ONE}[cost]\nsource_url = \"ftp://x/list.json\"\n"
        ));
        assert!(e.contains("cost.source_url: scheme"), "{e}");
        let e = err(&format!("{ONE}[cost]\nsource_url = \"nope\"\n"));
        assert!(e.contains("cost.source_url"), "{e}");
        let e = err(&format!("{ONE}[cost]\nrefresh = \"0s\"\n"));
        assert!(e.contains("cost.refresh"), "{e}");
    }

    const ONE: &str =
        "[[upstreams]]\nname = \"a\"\nprovider = \"openai\"\nbase_url = \"https://x\"\n\n";

    #[test]
    fn capture_parses_with_defaults() {
        let cfg = Config::parse(EXAMPLE).unwrap();
        let p = cfg.capture.unwrap();
        assert_eq!(p.max_bytes, 8 * 1024 * 1024);
        assert_eq!(p.retention, Duration::from_hours(30 * 24));
        assert_eq!(p.rules.len(), 2);
        assert!(p.rules[0].allow_header_override);
        assert_eq!(p.rules[0].matcher.tags["project"], "gateway-dev");
        assert!((p.rules[1].sample - 0.25).abs() < f64::EPSILON);
        assert!(
            p.rules[1]
                .matcher
                .model
                .as_ref()
                .unwrap()
                .is_match("claude-haiku-4-5")
        );

        let cfg = Config::parse(&format!(
            "{ONE}[store]\npath = \"g.db\"\n\n[capture]\nmax_bytes = \"1 KiB\"\nretention = \"2h\"\n\n[[capture.rules]]\nmatch = {{ tags = {{ Project = \"p\" }} }}\n"
        ))
        .unwrap();
        let p = cfg.capture.unwrap();
        assert_eq!(p.max_bytes, 1024);
        assert_eq!(p.retention, Duration::from_hours(2));
        assert!(p.rules[0].capture);
        assert!(p.rules[0].matcher.tags.contains_key("project"));
    }

    #[test]
    fn otel_parses_with_defaults() {
        let o = Config::parse(&format!("{ONE}[otel]\nendpoint = \"http://lgtm:4318\"\n"))
            .unwrap()
            .otel
            .unwrap();
        assert_eq!(o.traces_url.as_str(), "http://lgtm:4318/v1/traces");
        assert_eq!(o.service_name, "inference-gateway");
        assert!((o.sample_ratio - 1.0).abs() < f64::EPSILON);
        assert_eq!(o.flush, Duration::from_secs(2));
        let o = Config::parse(&format!(
            "{ONE}[otel]\nendpoint = \"http://c/otlp\"\nsample_ratio = 0.5\nflush = \"5s\"\n"
        ))
        .unwrap()
        .otel
        .unwrap();
        assert_eq!(o.traces_url.as_str(), "http://c/otlp/v1/traces");
        assert_eq!(o.flush, Duration::from_secs(5));
    }

    #[test]
    fn otel_errors_name_the_key() {
        let e = err(&format!("{ONE}[otel]\nendpoint = \"grpc://c\"\n"));
        assert!(e.contains("otel.endpoint"), "{e}");
        let e = err(&format!(
            "{ONE}[otel]\nendpoint = \"http://c\"\nsample_ratio = 2.0\n"
        ));
        assert!(e.contains("otel.sample_ratio"), "{e}");
    }

    #[test]
    fn capture_needs_store() {
        let e = err(&format!("{ONE}[capture]\n"));
        assert!(e.contains("[capture] needs a [store]"), "{e}");
    }

    #[test]
    fn capture_rule_errors_name_the_rule() {
        let store = "[store]\npath = \"g.db\"\n\n[capture]\n";
        let rule = |body: &str| {
            err(&format!(
                "{ONE}{store}[[capture.rules]]\ncapture = true\n\n[[capture.rules]]\n{body}\n"
            ))
        };
        let e = rule("match = { upstream = \"nope\" }");
        assert!(e.contains("capture.rules[1]: match.upstream"), "{e}");
        let e = rule("match = { model = \"a[\" }");
        assert!(e.contains("capture.rules[1]: match.model"), "{e}");
        let e = rule("sample = 1.5");
        assert!(e.contains("capture.rules[1]: sample"), "{e}");
        let e = rule("match = { stop_reason = [\"end_turn\"] }");
        assert!(
            e.contains("capture.rules[1]: match.stop_reason is not known"),
            "{e}"
        );
        let e = rule("match = { structured_output = false }");
        assert!(
            e.contains("capture.rules[1]: match.structured_output is for evaluators only"),
            "{e}"
        );
    }

    #[test]
    fn capture_type_error_reports_line_and_column() {
        let e = err(&format!(
            "{ONE}[store]\npath = \"g.db\"\n\n[capture]\nretention = \"soon\"\n"
        ));
        assert!(e.contains("line 10"), "{e}");
        assert!(e.contains("retention"), "{e}");
    }

    #[test]
    fn store_path_is_relative_to_config_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gw.toml");
        std::fs::write(&path, format!("{ONE}[store]\npath = \"gateway.db\"\n")).unwrap();
        let cfg = Config::load(&path).unwrap();
        assert_eq!(cfg.store.unwrap().path, dir.path().join("gateway.db"));
    }

    #[test]
    fn no_upstreams_is_an_error() {
        assert!(err("upstreams = []").contains("at least one"));
    }

    #[test]
    fn bad_cidr_is_an_error() {
        let e = err(r#"
[[upstreams]]
name = "a"
provider = "openai"
base_url = "https://x"

[ingress]
allow_cidrs = ["10.0.0.0/8", "nope"]
"#);
        assert!(e.contains("line 8"), "{e}");
        assert!(e.contains("nope"), "{e}");
    }
}
