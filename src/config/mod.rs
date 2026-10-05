//! Domain config: TOML parsed into [`FileConfig`], then validated into
//! [`Config`], the only form code past startup sees.

pub mod cli;
pub mod eval;

use std::collections::{BTreeMap, HashSet};
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
    /// Hand-maintained prices; win per model over `source_url`.
    pub pricing_file: Option<PathBuf>,
    /// Price list in `LiteLLM`'s `model_prices_and_context_window.json` format,
    /// fetched at startup and every `refresh`.
    pub source_url: Option<String>,
    #[serde(default = "default_price_refresh", with = "humantime_serde")]
    pub refresh: Duration,
    /// Last good download, used until the first fetch succeeds. Defaults to
    /// `pricing-cache.json` next to the config file.
    pub cache_file: Option<PathBuf>,
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
                if let Some(file) = &mut cost.pricing_file {
                    *file = dir.join(&*file);
                }
                if cost.source_url.is_some() {
                    let cache = cost.cache_file.take();
                    cost.cache_file =
                        Some(dir.join(cache.unwrap_or_else(|| DEFAULT_PRICE_CACHE.into())));
                }
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

        Ok(Self {
            upstreams,
            ingress: file.ingress,
            cost: file.cost,
            prometheus: file.prometheus,
            store: file.store,
            capture,
            eval,
        })
    }
}

fn validate_cost(cost: &CostFile) -> Result<()> {
    match &cost.source_url {
        Some(url) => {
            let url = Url::parse(url).context("cost.source_url")?;
            if !matches!(url.scheme(), "http" | "https") {
                bail!(
                    "cost.source_url: scheme must be http or https, got {:?}",
                    url.scheme()
                );
            }
            if cost.refresh.is_zero() {
                bail!("cost.refresh must not be zero");
            }
        }
        None if cost.pricing_file.is_none() => {
            bail!("[cost] needs pricing_file, source_url or both");
        }
        None if cost.cache_file.is_some() => {
            bail!("cost.cache_file needs source_url");
        }
        None => {}
    }
    Ok(())
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
    if name.is_empty() {
        bail!("must not be empty");
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
    fn pricing_file_is_relative_to_config_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gw.toml");
        std::fs::write(
            &path,
            "[[upstreams]]\nname = \"a\"\nprovider = \"openai\"\nbase_url = \"https://x\"\n\n[cost]\npricing_file = \"pricing.toml\"\n",
        )
        .unwrap();
        let cfg = Config::load(&path).unwrap();
        let cost = cfg.cost.unwrap();
        assert_eq!(cost.pricing_file, Some(dir.path().join("pricing.toml")));
        assert_eq!(cost.cache_file, None);
    }

    #[test]
    fn cost_source_url_gets_a_cache_file_next_to_the_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gw.toml");
        let write = |extra: &str| {
            std::fs::write(
                &path,
                format!("{ONE}[cost]\nsource_url = \"https://prices.example/list.json\"\n{extra}"),
            )
            .unwrap();
            Config::load(&path).unwrap().cost.unwrap()
        };
        let cost = write("");
        assert_eq!(cost.pricing_file, None);
        assert_eq!(cost.refresh, Duration::from_hours(6));
        assert_eq!(cost.cache_file, Some(dir.path().join("pricing-cache.json")));
        let cost = write("refresh = \"30m\"\ncache_file = \"p.json\"\n");
        assert_eq!(cost.refresh, Duration::from_mins(30));
        assert_eq!(cost.cache_file, Some(dir.path().join("p.json")));
    }

    #[test]
    fn cost_needs_a_price_source() {
        let e = err(&format!("{ONE}[cost]\n"));
        assert!(
            e.contains("[cost] needs pricing_file, source_url or both"),
            "{e}"
        );
        let e = err(&format!(
            "{ONE}[cost]\nsource_url = \"ftp://x/list.json\"\n"
        ));
        assert!(e.contains("cost.source_url: scheme"), "{e}");
        let e = err(&format!("{ONE}[cost]\nsource_url = \"nope\"\n"));
        assert!(e.contains("cost.source_url"), "{e}");
        let e = err(&format!(
            "{ONE}[cost]\npricing_file = \"p.toml\"\ncache_file = \"c.json\"\n"
        ));
        assert!(e.contains("cost.cache_file needs source_url"), "{e}");
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
