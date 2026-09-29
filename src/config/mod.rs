//! Domain config: TOML parsed into [`FileConfig`], then validated into
//! [`Config`], the only form code past startup sees.

pub mod cli;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use ipnet::IpNet;
use serde::Deserialize;
use url::Url;

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
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamFile {
    pub name: String,
    pub provider: ProviderKind,
    pub base_url: String,
    pub inject_stream_usage: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IngressFile {
    #[serde(default)]
    pub allow_cidrs: Vec<String>,
    #[serde(default)]
    pub require_token: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostFile {
    pub pricing_file: PathBuf,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrometheusFile {}

/// Module sections stay ungated here, so a section for a disabled feature
/// still parses and `modules::build` can name the missing feature.
#[derive(Debug, Clone)]
pub struct Config {
    pub upstreams: Vec<Upstream>,
    pub ingress: Ingress,
    pub cost: Option<CostFile>,
    pub prometheus: Option<PrometheusFile>,
}

#[derive(Debug, Clone)]
pub struct Upstream {
    pub name: String,
    pub provider: ProviderKind,
    pub base_url: Url,
    pub inject_stream_usage: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Ingress {
    pub allow_cidrs: Vec<IpNet>,
    pub require_token: bool,
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
        if let (Some(cost), Some(dir)) = (&mut config.cost, path.parent()) {
            cost.pricing_file = dir.join(&cost.pricing_file);
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

        let allow_cidrs = file
            .ingress
            .allow_cidrs
            .iter()
            .map(|c| {
                c.parse::<IpNet>()
                    .with_context(|| format!("ingress.allow_cidrs: {c:?} is not a CIDR"))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            upstreams,
            ingress: Ingress {
                allow_cidrs,
                require_token: file.ingress.require_token,
            },
            cost: file.cost,
            prometheus: file.prometheus,
        })
    }
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
        assert_eq!(
            cfg.cost.unwrap().pricing_file,
            dir.path().join("pricing.toml")
        );
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
        assert!(e.contains(r#"ingress.allow_cidrs: "nope""#), "{e}");
    }
}
