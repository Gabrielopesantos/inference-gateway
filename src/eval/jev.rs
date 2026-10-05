//! Jev decision model client: one question per call, typed answer back.
//! Wire format: <https://docs.typesafe.ai/api>.

use std::collections::BTreeMap;
use std::fmt;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use reqwest::StatusCode;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap};
use serde::Deserialize;
use serde_json::json;
use tokio::time::Instant;

use crate::config::eval::{Judge as JudgeConfig, Question};

/// Error bodies kept in `last_error`.
const MAX_ERROR_BODY: usize = 2048;

pub struct Judge {
    client: reqwest::Client,
    url: String,
    model: String,
    key: String,
    usd_per_mtok_input: f64,
}

impl fmt::Debug for Judge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Judge")
            .field("url", &self.url)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

/// A validated answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub verdict: Verdict,
    /// `None` for noul: Jev returns none.
    pub confidence: Option<f64>,
    pub distribution: BTreeMap<String, f64>,
    /// The version that answered, e.g. `jev-1.13.0` for `jev-latest`.
    pub resolved_model: String,
    pub input_tokens: u64,
    pub cost_usd: f64,
    pub latency: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Probability of yes.
    Noul(f64),
    Choice(String),
    /// Probability-weighted level index, and the most probable level's legend.
    Score {
        value: f64,
        label: String,
    },
}

impl Verdict {
    #[must_use]
    pub fn data_type(&self) -> &'static str {
        match self {
            Self::Noul(_) => "bool",
            Self::Choice(_) => "categorical",
            Self::Score { .. } => "numeric",
        }
    }

    #[must_use]
    pub fn value_bool(&self) -> Option<bool> {
        match self {
            Self::Noul(p) => Some(*p >= 0.5),
            _ => None,
        }
    }

    #[must_use]
    pub fn value_label(&self) -> Option<&str> {
        match self {
            Self::Noul(_) => None,
            Self::Choice(l) | Self::Score { label: l, .. } => Some(l),
        }
    }

    #[must_use]
    pub fn value_num(&self) -> Option<f64> {
        match self {
            Self::Noul(p) => Some(*p),
            Self::Choice(_) => None,
            Self::Score { value, .. } => Some(*value),
        }
    }

    /// Metric label: noul as `true` / `false`.
    #[must_use]
    pub fn label(&self) -> &str {
        match self.value_bool() {
            Some(true) => "true",
            Some(false) => "false",
            None => self.value_label().unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct JudgeError {
    /// Metric label: `http_<status>`, `timeout`, `connect`, `invalid`.
    pub reason: String,
    /// For `last_error`; never holds the key.
    pub message: String,
    pub retryable: bool,
    /// From `retry-after-ms` or `retry-after`.
    pub retry_after: Option<Duration>,
}

impl JudgeError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            reason: "invalid".into(),
            message: message.into(),
            retryable: true,
            retry_after: None,
        }
    }
}

impl fmt::Display for JudgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.reason, self.message)
    }
}

#[derive(Deserialize)]
struct Response {
    model: String,
    answers: BTreeMap<String, serde_json::Value>,
    usage: Usage,
}

#[derive(Deserialize)]
struct Usage {
    input_tokens: u64,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Answer {
    Noul {
        noul: f64,
    },
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
    Score {
        score: f64,
        #[serde(default)]
        legend: BTreeMap<String, String>,
        probabilities: BTreeMap<String, f64>,
        confidence: f64,
    },
}

impl Judge {
    /// # Errors
    ///
    /// HTTP client build failure.
    pub fn new(cfg: &JudgeConfig, key: String) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(cfg.timeout)
            .build()
            .context("building judge HTTP client")?;
        Ok(Self {
            client,
            url: format!(
                "{}/v1/systemone",
                cfg.base_url.as_str().trim_end_matches('/')
            ),
            model: cfg.model.clone(),
            key,
            usd_per_mtok_input: cfg.usd_per_mtok_input,
        })
    }

    /// Asks one question, named `name`, about `state`.
    ///
    /// # Errors
    ///
    /// Transport, HTTP or answer validation failure; see [`JudgeError::retryable`].
    pub async fn ask(
        &self,
        name: &str,
        state: &str,
        question: &Question,
    ) -> Result<Decision, JudgeError> {
        let body = json!({
            "model": self.model,
            "state": state,
            "questions": { name: question },
        });
        let started = Instant::now();
        let resp = self
            .client
            .post(&self.url)
            .header(AUTHORIZATION, format!("Bearer {}", self.key))
            .header(CONTENT_TYPE, "application/json")
            .body(body.to_string())
            .send()
            .await
            .map_err(transport)?;
        let status = resp.status();
        let retry_after = retry_after(resp.headers(), SystemTime::now());
        let bytes = resp.bytes().await.map_err(transport)?;
        let latency = started.elapsed();
        if !status.is_success() {
            let text = String::from_utf8_lossy(&bytes[..bytes.len().min(MAX_ERROR_BODY)]);
            return Err(JudgeError {
                reason: format!("http_{}", status.as_u16()),
                message: format!("{status}: {text}"),
                retryable: retryable(status),
                retry_after,
            });
        }
        let r: Response = serde_json::from_slice(&bytes)
            .map_err(|e| JudgeError::invalid(format!("response body: {e}")))?;
        let answer = r
            .answers
            .get(name)
            .ok_or_else(|| JudgeError::invalid(format!("no answer for {name:?}")))?;
        let (verdict, confidence, distribution) = validate(answer, question)?;
        #[allow(clippy::cast_precision_loss)]
        let cost_usd = r.usage.input_tokens as f64 * self.usd_per_mtok_input / 1e6;
        Ok(Decision {
            verdict,
            confidence,
            distribution,
            resolved_model: r.model,
            input_tokens: r.usage.input_tokens,
            cost_usd,
            latency,
        })
    }
}

/// 401 and 422 are config errors: retrying cannot help.
fn retryable(status: StatusCode) -> bool {
    status == StatusCode::REQUEST_TIMEOUT
        || status == StatusCode::TOO_MANY_REQUESTS
        || status.is_server_error()
}

fn transport(e: reqwest::Error) -> JudgeError {
    let reason = if e.is_timeout() {
        "timeout"
    } else if e.is_connect() {
        "connect"
    } else {
        "transport"
    };
    JudgeError {
        reason: reason.into(),
        // Without the URL, which is config, not news.
        message: e.without_url().to_string(),
        retryable: true,
        retry_after: None,
    }
}

/// `retry-after-ms` first, then `retry-after` in seconds or as an HTTP date.
fn retry_after(headers: &HeaderMap, now: SystemTime) -> Option<Duration> {
    let get = |name| headers.get(name).and_then(|v| v.to_str().ok());
    if let Some(ms) = get("retry-after-ms").and_then(|v| v.trim().parse::<f64>().ok())
        && ms.is_finite()
        && ms >= 0.0
    {
        return Some(Duration::from_secs_f64(ms / 1000.0));
    }
    let v = get("retry-after")?.trim();
    if let Ok(secs) = v.parse::<f64>() {
        return (secs.is_finite() && secs >= 0.0).then(|| Duration::from_secs_f64(secs));
    }
    httpdate::parse_http_date(v)
        .ok()
        .map(|at| at.duration_since(now).unwrap_or_default())
}

type Validated = (Verdict, Option<f64>, BTreeMap<String, f64>);

fn validate(raw: &serde_json::Value, question: &Question) -> Result<Validated, JudgeError> {
    let answer =
        Answer::deserialize(raw).map_err(|e| JudgeError::invalid(format!("answer: {e}")))?;
    let unit = |what: &str, v: f64| {
        if (0.0..=1.0).contains(&v) {
            Ok(())
        } else {
            Err(JudgeError::invalid(format!("{what} {v} outside [0, 1]")))
        }
    };
    let probabilities = |p: &BTreeMap<String, f64>, expected: Vec<String>| {
        let mut keys: Vec<_> = p.keys().cloned().collect();
        let mut expected = expected;
        keys.sort();
        expected.sort();
        if keys != expected {
            return Err(JudgeError::invalid(format!(
                "probability keys {keys:?}, expected {expected:?}"
            )));
        }
        p.iter()
            .try_for_each(|(k, v)| unit(&format!("probability of {k:?}"), *v))
    };
    match (answer, question) {
        (Answer::Noul { noul }, Question::Noul { .. }) => {
            unit("noul", noul)?;
            let dist = BTreeMap::from([("true".into(), noul), ("false".into(), 1.0 - noul)]);
            Ok((Verdict::Noul(noul), None, dist))
        }
        (
            Answer::Choice {
                choice,
                probabilities: p,
                confidence,
            },
            Question::Choice { criteria, .. },
        ) => {
            if !criteria.contains_key(&choice) {
                return Err(JudgeError::invalid(format!(
                    "choice {choice:?} not in criteria"
                )));
            }
            probabilities(&p, criteria.keys().cloned().collect())?;
            unit("confidence", confidence)?;
            Ok((Verdict::Choice(choice), Some(confidence), p))
        }
        (
            Answer::Score {
                score,
                legend,
                probabilities: p,
                confidence,
            },
            Question::Score { criteria, .. },
        ) => {
            #[allow(clippy::cast_precision_loss)]
            let top = (criteria.len() - 1) as f64;
            if !(0.0..=top).contains(&score) {
                return Err(JudgeError::invalid(format!(
                    "score {score} outside [0, {top}]"
                )));
            }
            probabilities(&p, (0..criteria.len()).map(|i| i.to_string()).collect())?;
            unit("confidence", confidence)?;
            let (best, _) = p
                .iter()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .ok_or_else(|| JudgeError::invalid("no probabilities"))?;
            let label = legend.get(best).cloned().unwrap_or_else(|| {
                best.parse::<usize>()
                    .ok()
                    .and_then(|i| criteria.get(i).cloned())
                    .unwrap_or_default()
            });
            Ok((
                Verdict::Score {
                    value: score,
                    label,
                },
                Some(confidence),
                p,
            ))
        }
        (a, q) => Err(JudgeError::invalid(format!(
            "answer type {} for a {} question",
            match a {
                Answer::Noul { .. } => "noul",
                Answer::Choice { .. } => "choice",
                Answer::Score { .. } => "score",
            },
            q.kind()
        ))),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use serde_json::Value;

    use super::*;

    fn fixture(name: &str) -> Value {
        let path = format!(
            "{}/tests/fixtures/jev/{name}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn question(toml_text: &str) -> Question {
        toml::from_str(toml_text).unwrap()
    }

    fn choice() -> Question {
        question(
            "type = \"choice\"\ninstructions = \"i\"\n[criteria]\ncorrect = \"a\"\npartial = \"b\"\nwrong = \"\"\n",
        )
    }

    fn score() -> Question {
        question(
            "type = \"score\"\ninstructions = \"i\"\ncriteria = [\"bad\", \"ok\", \"great\"]\n",
        )
    }

    fn noul() -> Question {
        question("type = \"noul\"\ninstructions = \"i\"\n")
    }

    fn answer(fixture_name: &str) -> Value {
        fixture(fixture_name)["answers"]["q"].clone()
    }

    #[test]
    fn choice_maps_label_confidence_and_distribution() {
        let (v, conf, dist) = validate(&answer("choice"), &choice()).unwrap();
        assert_eq!(v, Verdict::Choice("correct".into()));
        assert_eq!(v.data_type(), "categorical");
        assert_eq!(conf, Some(0.95));
        assert_eq!(dist.len(), 3);
    }

    #[test]
    fn noul_maps_to_bool_and_probability() {
        let (v, conf, dist) = validate(&answer("noul"), &noul()).unwrap();
        assert_eq!(v.value_bool(), Some(true));
        assert_eq!(v.value_num(), Some(0.99));
        assert_eq!(v.label(), "true");
        assert_eq!(conf, None);
        assert!((dist["false"] - 0.01).abs() < 1e-9);
    }

    #[test]
    fn score_maps_value_and_most_probable_legend() {
        let (v, conf, _) = validate(&answer("score"), &score()).unwrap();
        assert_eq!(
            v,
            Verdict::Score {
                value: 1.37,
                label: "ok".into()
            }
        );
        assert_eq!(v.data_type(), "numeric");
        assert_eq!(conf, Some(0.44));
    }

    #[test]
    fn invalid_answers_are_retryable() {
        let mut a = answer("choice");
        a["choice"] = "maybe".into();
        let e = validate(&a, &choice()).unwrap_err();
        assert!(e.retryable && e.reason == "invalid", "{e}");
        assert!(e.message.contains("not in criteria"), "{e}");

        let mut a = answer("choice");
        a["probabilities"].as_object_mut().unwrap().remove("wrong");
        assert!(
            validate(&a, &choice())
                .unwrap_err()
                .message
                .contains("keys")
        );

        let mut a = answer("choice");
        a["confidence"] = 1.5.into();
        assert!(
            validate(&a, &choice())
                .unwrap_err()
                .message
                .contains("confidence")
        );

        let mut a = answer("score");
        a["score"] = 3.0.into();
        assert!(
            validate(&a, &score())
                .unwrap_err()
                .message
                .contains("outside")
        );

        let e = validate(&answer("noul"), &choice()).unwrap_err();
        assert!(e.message.contains("answer type noul for a choice"), "{e}");

        let e = validate(&serde_json::json!({"type": "noul"}), &noul()).unwrap_err();
        assert!(e.retryable);
    }

    #[test]
    fn retry_after_headers() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000_000);
        let h = |pairs: &[(&'static str, &str)]| {
            let mut map = HeaderMap::new();
            for (k, v) in pairs {
                map.insert(*k, v.parse().unwrap());
            }
            retry_after(&map, now)
        };
        assert_eq!(h(&[]), None);
        assert_eq!(h(&[("retry-after", "2")]), Some(Duration::from_secs(2)));
        assert_eq!(
            h(&[("retry-after", "2"), ("retry-after-ms", "1500")]),
            Some(Duration::from_millis(1500))
        );
        let date = httpdate::fmt_http_date(now + Duration::from_secs(30));
        assert_eq!(h(&[("retry-after", &date)]), Some(Duration::from_secs(30)));
        assert_eq!(h(&[("retry-after", "soon")]), None);
    }

    #[test]
    fn status_classes() {
        for s in [408, 429, 500, 503, 529] {
            assert!(retryable(StatusCode::from_u16(s).unwrap()), "{s}");
        }
        for s in [400, 401, 403, 404, 422] {
            assert!(!retryable(StatusCode::from_u16(s).unwrap()), "{s}");
        }
    }
}
