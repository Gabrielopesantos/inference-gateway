//! Which requests get their content captured. Decided once per request,
//! before forwarding, from ordered rules; the first matching rule wins.

use std::collections::BTreeMap;
use std::time::Duration;

use globset::GlobMatcher;
use sha2::{Digest, Sha256};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Policy {
    /// Bound on the raw-response copy; the request body is bounded by `--max-body`.
    pub max_bytes: usize,
    /// Age after which content is deleted; trace rows are kept.
    pub retention: Duration,
    pub rules: Vec<Rule>,
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub upstream: Option<String>,
    pub model: Option<GlobMatcher>,
    /// Every pair must be present on the request.
    pub tags: BTreeMap<String, String>,
    pub capture: bool,
    /// Fraction of matching requests captured, in `[0, 1]`.
    pub sample: f64,
    /// Lets `x-gateway-capture: 0|1` override `capture` and `sample`.
    pub allow_header_override: bool,
}

impl Rule {
    fn matches(
        &self,
        upstream: &str,
        model: Option<&str>,
        tags: &BTreeMap<String, String>,
    ) -> bool {
        self.upstream.as_deref().is_none_or(|u| u == upstream)
            && self
                .model
                .as_ref()
                .is_none_or(|g| model.is_some_and(|m| g.is_match(m)))
            && self.tags.iter().all(|(k, v)| tags.get(k) == Some(v))
    }
}

impl Policy {
    /// No matching rule means no capture.
    #[must_use]
    pub fn decide(
        &self,
        upstream: &str,
        model: Option<&str>,
        tags: &BTreeMap<String, String>,
        header: Option<bool>,
        trace_id: Uuid,
    ) -> bool {
        let Some(rule) = self.rules.iter().find(|r| r.matches(upstream, model, tags)) else {
            return false;
        };
        if rule.allow_header_override
            && let Some(h) = header
        {
            return h;
        }
        rule.capture && sampled(trace_id, rule.sample)
    }
}

/// Deterministic per trace. Hashed because `UUIDv7` bits are time-ordered.
fn sampled(trace_id: Uuid, rate: f64) -> bool {
    if rate >= 1.0 {
        return true;
    }
    let digest = Sha256::digest(trace_id.as_bytes());
    let mut head = [0; 8];
    head.copy_from_slice(&digest[..8]);
    #[allow(clippy::cast_precision_loss)]
    let point = u64::from_be_bytes(head) as f64 / u64::MAX as f64;
    point < rate
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn rule() -> Rule {
        Rule {
            upstream: None,
            model: None,
            tags: BTreeMap::new(),
            capture: true,
            sample: 1.0,
            allow_header_override: false,
        }
    }

    fn policy(rules: Vec<Rule>) -> Policy {
        Policy {
            max_bytes: 1024,
            retention: Duration::from_mins(1),
            rules,
        }
    }

    fn tags(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect()
    }

    fn glob(p: &str) -> GlobMatcher {
        globset::Glob::new(p).unwrap().compile_matcher()
    }

    #[test]
    fn no_rules_means_no_capture() {
        let p = policy(vec![]);
        assert!(!p.decide("a", Some("m"), &tags(&[]), Some(true), Uuid::now_v7()));
    }

    #[test]
    fn first_match_wins() {
        let p = policy(vec![
            Rule {
                model: Some(glob("claude-haiku-*")),
                capture: false,
                ..rule()
            },
            rule(),
        ]);
        let id = Uuid::now_v7();
        let none = tags(&[]);
        assert!(!p.decide("a", Some("claude-haiku-4-5"), &none, None, id));
        assert!(p.decide("a", Some("claude-sonnet-5"), &none, None, id));
        // A model glob never matches a request without a model.
        assert!(p.decide("a", None, &none, None, id));
    }

    #[test]
    fn upstream_and_tags_must_all_match() {
        let p = policy(vec![Rule {
            upstream: Some("anthropic".into()),
            tags: tags(&[("project", "gw"), ("env", "dev")]),
            ..rule()
        }]);
        let id = Uuid::now_v7();
        let both = tags(&[("project", "gw"), ("env", "dev"), ("extra", "x")]);
        assert!(p.decide("anthropic", None, &both, None, id));
        assert!(!p.decide("openai", None, &both, None, id));
        assert!(!p.decide("anthropic", None, &tags(&[("project", "gw")]), None, id));
        assert!(!p.decide(
            "anthropic",
            None,
            &tags(&[("project", "gw"), ("env", "prod")]),
            None,
            id
        ));
    }

    #[test]
    fn header_overrides_only_when_allowed() {
        let id = Uuid::now_v7();
        let none = tags(&[]);
        let strict = policy(vec![Rule {
            capture: false,
            ..rule()
        }]);
        assert!(!strict.decide("a", None, &none, Some(true), id));

        let open = policy(vec![Rule {
            capture: false,
            sample: 0.0,
            allow_header_override: true,
            ..rule()
        }]);
        assert!(open.decide("a", None, &none, Some(true), id));
        assert!(!open.decide("a", None, &none, None, id));

        let on = policy(vec![Rule {
            allow_header_override: true,
            ..rule()
        }]);
        assert!(!on.decide("a", None, &none, Some(false), id));
        assert!(on.decide("a", None, &none, None, id));
    }

    #[test]
    fn sampling_is_deterministic_and_near_rate() {
        let p = policy(vec![Rule {
            sample: 0.25,
            ..rule()
        }]);
        let none = tags(&[]);
        // Fixed ids: random ones fail the ±1% bound about 2% of runs.
        let ids: Vec<_> = (0..10_000u128)
            .map(|i| Uuid::from_u128(i << 64 | 0x7000))
            .collect();
        let hits = ids
            .iter()
            .filter(|id| p.decide("a", None, &none, None, **id))
            .count();
        assert!((2400..=2600).contains(&hits), "{hits}");
        for id in &ids[..100] {
            assert_eq!(
                p.decide("a", None, &none, None, *id),
                p.decide("a", None, &none, None, *id)
            );
        }
    }

    #[test]
    fn sample_bounds() {
        let id = Uuid::now_v7();
        assert!(!sampled(id, 0.0));
        assert!(sampled(id, 1.0));
    }
}
