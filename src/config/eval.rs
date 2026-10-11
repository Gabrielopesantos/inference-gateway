//! `[eval]`: the worker, the judge and the evaluators that score captured
//! traces.

use std::collections::{BTreeMap, HashSet};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize, Serializer};
use sha2::{Digest, Sha256};
use url::Url;

use super::{MatchFile, Upstream, parse_base_url, parse_match, validate_name};
use crate::capture::Match;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalFile {
    #[serde(default)]
    pub worker: WorkerFile,
    pub judge: JudgeFile,
    #[serde(default)]
    pub evaluators: Vec<EvaluatorFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerFile {
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    #[serde(default = "default_max_attempts")]
    pub max_attempts: u32,
    #[serde(default = "default_lease", with = "humantime_serde")]
    pub lease: Duration,
    #[serde(default = "default_poll", with = "humantime_serde")]
    pub poll: Duration,
}

impl Default for WorkerFile {
    fn default() -> Self {
        Self {
            concurrency: default_concurrency(),
            max_attempts: default_max_attempts(),
            lease: default_lease(),
            poll: default_poll(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeFile {
    pub base_url: String,
    pub model: String,
    pub api_key_env: String,
    #[serde(default = "default_max_input_tokens")]
    pub max_input_tokens: usize,
    #[serde(default = "default_timeout", with = "humantime_serde")]
    pub timeout: Duration,
    #[serde(default = "default_usd_per_mtok_input")]
    pub usd_per_mtok_input: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorFile {
    pub name: String,
    #[serde(rename = "match", default)]
    pub matcher: MatchFile,
    #[serde(default = "super::default_sample")]
    pub sample: f64,
    pub input: String,
    pub kind: Kind,
    pub question: String,
    pub criteria: Option<CriteriaFile>,
    pub pass: Option<Vec<String>>,
    pub min_confidence: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    YesNo,
    Choice,
    Score,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum CriteriaFile {
    /// `yes_no` and `choice`: description per label.
    Labels(BTreeMap<String, String>),
    /// score: levels, lowest first.
    Levels(Vec<String>),
}

/// Judge wire shape, sent as is; also hashed into the evaluator version.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Probability of yes.
    #[serde(rename = "noul")]
    YesNo {
        instructions: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<YesNoCriteria>,
    },
    /// One label out of `criteria`.
    Choice {
        instructions: String,
        /// Description per label; `""` is sent as `null` (TOML has no null).
        #[serde(serialize_with = "empty_as_null")]
        criteria: BTreeMap<String, String>,
    },
    /// Level index into `criteria`, lowest first.
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct YesNoCriteria {
    #[serde(rename = "true", skip_serializing_if = "Option::is_none")]
    pub yes: Option<String>,
    #[serde(rename = "false", skip_serializing_if = "Option::is_none")]
    pub no: Option<String>,
}

fn empty_as_null<S: Serializer>(map: &BTreeMap<String, String>, s: S) -> Result<S::Ok, S::Error> {
    s.collect_map(map.iter().map(|(k, v)| (k, (!v.is_empty()).then_some(v))))
}

impl Question {
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::YesNo { .. } => "yes_no",
            Self::Choice { .. } => "choice",
            Self::Score { .. } => "score",
        }
    }
}

fn default_concurrency() -> usize {
    4
}

fn default_max_attempts() -> u32 {
    5
}

fn default_lease() -> Duration {
    Duration::from_mins(5)
}

fn default_poll() -> Duration {
    Duration::from_secs(1)
}

/// Typesafe Jev's limit: input plus the longest question.
fn default_max_input_tokens() -> usize {
    32_000
}

fn default_timeout() -> Duration {
    Duration::from_mins(1)
}

/// Typesafe Jev's list price; output tokens are not billed.
fn default_usd_per_mtok_input() -> f64 {
    0.042
}

#[derive(Debug, Clone)]
pub struct EvalConfig {
    pub worker: Worker,
    pub judge: Judge,
    pub evaluators: Vec<Evaluator>,
}

#[derive(Debug, Clone)]
pub struct Worker {
    /// Max jobs in flight, which is also max concurrent judge calls.
    pub concurrency: usize,
    pub max_attempts: u32,
    /// How long a claimed job is held before another worker may take it.
    pub lease: Duration,
    /// Queue poll interval; new jobs from this process wake the worker sooner.
    pub poll: Duration,
}

#[derive(Debug, Clone)]
pub struct Judge {
    pub base_url: Url,
    /// Pin a version: an alias such as `jev-latest` changes scores under
    /// the same evaluator version.
    pub model: String,
    /// Env var holding the key; read when the worker starts.
    pub api_key_env: String,
    pub max_input_tokens: usize,
    pub timeout: Duration,
    pub usd_per_mtok_input: f64,
}

#[derive(Debug, Clone)]
pub struct Evaluator {
    pub name: String,
    pub matcher: Match,
    /// Fraction of matching traces scored, in `[0, 1]`.
    pub sample: f64,
    /// minijinja template; compiled at validation.
    pub input: String,
    pub question: Question,
    /// Score labels that pass, as stored (`yes_no`: `yes`/`no`); `None`
    /// gives no verdict. Not part of the version.
    pub pass: Option<Vec<String>>,
    /// Below it the verdict is `uncertain`; every score is stored.
    pub min_confidence: Option<f64>,
    /// Changes with the judge model, input template or question.
    pub version: String,
}

impl Evaluator {
    /// Deterministic per trace and evaluator, so backfill picks the same
    /// traces as live enqueue.
    #[must_use]
    pub fn sampled(&self, trace_id: &[u8]) -> bool {
        let key = [trace_id, self.name.as_bytes()].concat();
        crate::capture::sampled(&key, self.sample)
    }
}

pub(super) fn eval_config(file: EvalFile, upstreams: &[Upstream]) -> Result<EvalConfig> {
    let w = file.worker;
    if w.concurrency == 0 {
        bail!("eval.worker.concurrency must be at least 1");
    }
    if w.max_attempts == 0 {
        bail!("eval.worker.max_attempts must be at least 1");
    }
    let j = file.judge;
    let judge = Judge {
        base_url: parse_base_url(&j.base_url).context("eval.judge.base_url")?,
        model: j.model,
        api_key_env: j.api_key_env,
        max_input_tokens: j.max_input_tokens,
        timeout: j.timeout,
        usd_per_mtok_input: j.usd_per_mtok_input,
    };
    if !super::valid_price(judge.usd_per_mtok_input) {
        bail!("eval.judge.usd_per_mtok_input must be finite and not negative");
    }
    if judge.api_key_env.is_empty() {
        bail!("eval.judge.api_key_env must not be empty");
    }
    let mut seen = HashSet::new();
    let evaluators = file
        .evaluators
        .into_iter()
        .enumerate()
        .map(|(i, e)| {
            let name = e.name.clone();
            if !seen.insert(name.clone()) {
                bail!("eval.evaluators[{i}] {name:?}: duplicate name");
            }
            evaluator(e, &judge.model, upstreams)
                .with_context(|| format!("eval.evaluators[{i}] {name:?}"))
        })
        .collect::<Result<_>>()?;
    Ok(EvalConfig {
        worker: Worker {
            concurrency: w.concurrency,
            max_attempts: w.max_attempts,
            lease: w.lease,
            poll: w.poll,
        },
        judge,
        evaluators,
    })
}

fn evaluator(e: EvaluatorFile, model: &str, upstreams: &[Upstream]) -> Result<Evaluator> {
    validate_name(&e.name).context("name")?;
    let matcher = parse_match(e.matcher, upstreams)?;
    if !(0.0..=1.0).contains(&e.sample) {
        bail!("sample must be in [0, 1], got {}", e.sample);
    }
    if let Some(c) = e.min_confidence
        && !(0.0..=1.0).contains(&c)
    {
        bail!("min_confidence must be in [0, 1], got {c}");
    }
    let question = question(e.kind, e.question, e.criteria).context("criteria")?;
    minijinja::Environment::new()
        .template_from_str(&e.input)
        .context("input template")?;
    let pass = e
        .pass
        .map(|p| pass_labels(p, &question))
        .transpose()
        .context("pass")?;
    let version = version(model, &e.input, &question)?;
    Ok(Evaluator {
        name: e.name,
        matcher,
        sample: e.sample,
        input: e.input,
        question,
        pass,
        min_confidence: e.min_confidence,
        version,
    })
}

fn question(kind: Kind, instructions: String, criteria: Option<CriteriaFile>) -> Result<Question> {
    Ok(match (kind, criteria) {
        (Kind::YesNo, None) => Question::YesNo {
            instructions,
            criteria: None,
        },
        (Kind::YesNo, Some(CriteriaFile::Labels(mut m))) => {
            let (yes, no) = (m.remove("yes"), m.remove("no"));
            if let Some(k) = m.keys().next() {
                bail!("yes_no takes yes and no, got {k:?}");
            }
            Question::YesNo {
                instructions,
                criteria: Some(YesNoCriteria { yes, no }),
            }
        }
        (Kind::Choice, Some(CriteriaFile::Labels(criteria))) => {
            if !(1..=255).contains(&criteria.len()) {
                bail!("choice needs 1 to 255 criteria, got {}", criteria.len());
            }
            Question::Choice {
                instructions,
                criteria,
            }
        }
        (Kind::Score, Some(CriteriaFile::Levels(criteria))) => {
            if !(2..=10).contains(&criteria.len()) {
                bail!("score needs 2 to 10 criteria, got {}", criteria.len());
            }
            Question::Score {
                instructions,
                criteria,
            }
        }
        (Kind::YesNo, Some(CriteriaFile::Levels(_))) => bail!("yes_no takes a table"),
        (Kind::Choice, _) => bail!("choice needs a table"),
        (Kind::Score, _) => bail!("score needs a list of levels"),
    })
}

fn pass_labels(labels: Vec<String>, question: &Question) -> Result<Vec<String>> {
    if labels.is_empty() {
        bail!("empty list");
    }
    for l in &labels {
        let known = match question {
            Question::YesNo { .. } => l == "yes" || l == "no",
            Question::Choice { criteria, .. } => criteria.contains_key(l),
            Question::Score { criteria, .. } => criteria.contains(l),
        };
        if !known {
            bail!("{l:?} is not a {} label", question.kind());
        }
    }
    Ok(labels)
}

fn version(model: &str, input: &str, question: &Question) -> Result<String> {
    let mut h = Sha256::new();
    for part in [model, input, &serde_json::to_string(question)?] {
        // Length-prefixed, so moving text between parts changes the hash.
        h.update((part.len() as u64).to_be_bytes());
        h.update(part);
    }
    Ok(hex::encode(&h.finalize()[..8]))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::Config;
    use super::*;

    const BASE: &str = r#"
[[upstreams]]
name = "a"
provider = "anthropic"
base_url = "https://x"

[store]
path = "g.db"

[eval.judge]
base_url = "https://api.typesafe.ai"
model = "jev-1.13.0"
api_key_env = "IG_JUDGE_API_KEY"
"#;

    const CHOICE: &str = r#"
[[eval.evaluators]]
name = "answered"
input = "<user>{{ last_user_message }}</user>"
kind = "choice"
question = "Did it answer?"
criteria = { yes = "Fully", no = "" }
"#;

    fn parse(extra: &str) -> Result<EvalConfig> {
        Config::parse(&format!("{BASE}{extra}")).map(|c| c.eval.unwrap())
    }

    fn err(extra: &str) -> String {
        format!("{:#}", parse(extra).unwrap_err())
    }

    fn one(fields: &str) -> String {
        format!("[[eval.evaluators]]\nname = \"e\"\ninput = \"x\"\nquestion = \"i\"\n{fields}\n")
    }

    #[test]
    fn parses_with_defaults() {
        let c = parse(CHOICE).unwrap();
        assert_eq!(c.worker.concurrency, 4);
        assert_eq!(c.worker.max_attempts, 5);
        assert_eq!(c.worker.lease, Duration::from_mins(5));
        assert_eq!(c.judge.max_input_tokens, 32_000);
        let e = &c.evaluators[0];
        assert_eq!(e.question.kind(), "choice");
        assert!((e.sample - 1.0).abs() < f64::EPSILON);
        assert_eq!(e.version.len(), 16);
    }

    #[test]
    fn worker_section_overrides_defaults() {
        let c = parse(&format!(
            "[eval.worker]\nconcurrency = 2\nlease = \"1m\"\n{CHOICE}"
        ))
        .unwrap();
        assert_eq!(c.worker.concurrency, 2);
        assert_eq!(c.worker.lease, Duration::from_mins(1));
        assert_eq!(c.worker.max_attempts, 5);
        let e = err(&format!("[eval.worker]\nconcurrency = 0\n{CHOICE}"));
        assert!(e.contains("eval.worker.concurrency"), "{e}");
    }

    #[test]
    fn question_serializes_to_the_wire_shape() {
        let c = parse(CHOICE).unwrap();
        let wire = serde_json::to_value(&c.evaluators[0].question).unwrap();
        assert_eq!(
            wire,
            serde_json::json!({
                "type": "choice",
                "instructions": "Did it answer?",
                "criteria": {"no": null, "yes": "Fully"},
            })
        );
        let c = parse(&one("kind = \"yes_no\"\ncriteria = { yes = \"y\" }")).unwrap();
        assert_eq!(
            serde_json::to_value(&c.evaluators[0].question).unwrap(),
            serde_json::json!({"type": "noul", "instructions": "i", "criteria": {"true": "y"}})
        );
        let c = parse(&one("kind = \"yes_no\"")).unwrap();
        assert_eq!(
            serde_json::to_value(&c.evaluators[0].question).unwrap(),
            serde_json::json!({"type": "noul", "instructions": "i"})
        );
    }

    #[test]
    fn eval_needs_store() {
        let text = BASE.replace("[store]\npath = \"g.db\"\n", "");
        let e = format!("{:#}", Config::parse(&text).unwrap_err());
        assert!(e.contains("[eval] needs a [store]"), "{e}");
    }

    #[test]
    fn validation_errors_name_the_evaluator() {
        let e = err(&format!("{CHOICE}{CHOICE}"));
        assert!(
            e.contains(r#"eval.evaluators[1] "answered": duplicate name"#),
            "{e}"
        );

        let eleven = format!("kind = \"score\"\ncriteria = {:?}", vec!["l"; 11]);
        for (fields, msg) in [
            (
                "kind = \"score\"\ncriteria = [\"one\"]",
                "score needs 2 to 10",
            ),
            (eleven.as_str(), "score needs 2 to 10"),
            (
                "kind = \"score\"\ncriteria = { a = \"\" }",
                "score needs a list",
            ),
            ("kind = \"score\"", "score needs a list"),
            ("kind = \"choice\"\ncriteria = {}", "choice needs 1 to 255"),
            (
                "kind = \"choice\"\ncriteria = [\"a\"]",
                "choice needs a table",
            ),
            ("kind = \"choice\"", "choice needs a table"),
            (
                "kind = \"yes_no\"\ncriteria = { true = \"\" }",
                r#"yes_no takes yes and no, got "true""#,
            ),
            (
                "kind = \"yes_no\"\ncriteria = [\"a\"]",
                "yes_no takes a table",
            ),
        ] {
            let e = err(&one(fields));
            assert!(
                e.contains(&format!(r#"eval.evaluators[0] "e": criteria: {msg}"#)),
                "{fields}: {e}"
            );
        }

        let e = err(&CHOICE.replace("{{ last_user_message }}", "{{ unclosed"));
        assert!(e.contains("input template"), "{e}");

        let e = err(&CHOICE.replace("name = \"answered\"", "name = \"a b\""));
        assert!(e.contains("name"), "{e}");

        let e = err(&CHOICE.replace(
            "name = \"answered\"\n",
            "name = \"answered\"\nmatch = { upstream = \"nope\" }\n",
        ));
        assert!(e.contains("match.upstream"), "{e}");
    }

    #[test]
    fn pass_labels_are_checked() {
        let with = |pass: &str| {
            parse(&CHOICE.replace(
                "name = \"answered\"\n",
                &format!("name = \"answered\"\npass = {pass}\n"),
            ))
        };
        let c = with("[\"yes\"]").unwrap();
        assert_eq!(c.evaluators[0].pass, Some(vec!["yes".to_owned()]));
        assert_eq!(parse(CHOICE).unwrap().evaluators[0].pass, None);
        assert_eq!(
            c.evaluators[0].version,
            parse(CHOICE).unwrap().evaluators[0].version
        );

        for (pass, msg) in [
            ("[\"maybe\"]", r#""maybe" is not a choice label"#),
            ("[]", "empty list"),
        ] {
            let e = format!("{:#}", with(pass).unwrap_err());
            assert!(e.contains(&format!("pass: {msg}")), "{pass}: {e}");
        }
        assert!(format!("{:#}", with("\"yes\"").unwrap_err()).contains("pass"));

        let yes_no = |pass: &str| parse(&one(&format!("kind = \"yes_no\"\npass = {pass}")));
        assert_eq!(
            yes_no("[\"no\"]").unwrap().evaluators[0].pass,
            Some(vec!["no".to_owned()])
        );
        let e = format!("{:#}", yes_no("[\"true\"]").unwrap_err());
        assert!(e.contains(r#""true" is not a yes_no label"#), "{e}");

        let score = parse(&one(
            "kind = \"score\"\ncriteria = [\"bad\", \"good\"]\npass = [\"good\"]",
        ))
        .unwrap();
        assert_eq!(score.evaluators[0].pass, Some(vec!["good".to_owned()]));
    }

    #[test]
    fn unknown_kind_reports_line() {
        let e = err(&CHOICE.replace("kind = \"choice\"", "kind = \"rank\""));
        assert!(e.contains("rank"), "{e}");
        assert!(e.contains("line"), "{e}");
    }

    #[test]
    fn version_tracks_model_template_and_question() {
        let base = parse(CHOICE).unwrap().evaluators[0].version.clone();
        assert_eq!(parse(CHOICE).unwrap().evaluators[0].version, base);
        let changed = [
            parse(&CHOICE.replace("<user>", "<u>")).unwrap(),
            parse(&CHOICE.replace("Did it answer?", "Answered?")).unwrap(),
            parse(&CHOICE.replace("no = \"\"", "no = \"Not at all\"")).unwrap(),
            Config::parse(&format!(
                "{}{CHOICE}",
                BASE.replace("jev-1.13.0", "jev-1.14.0")
            ))
            .unwrap()
            .eval
            .unwrap(),
        ];
        for c in changed {
            assert_ne!(c.evaluators[0].version, base);
        }
        // Not part of the version: which traces get scored.
        for selection in ["sample = 0.5", "match = { stop_reason = [\"end_turn\"] }"] {
            let c = parse(&CHOICE.replace(
                "name = \"answered\"\n",
                &format!("name = \"answered\"\n{selection}\n"),
            ))
            .unwrap();
            assert_eq!(c.evaluators[0].version, base, "{selection}");
        }
    }

    #[test]
    fn sampling_differs_per_evaluator() {
        let mut c = parse(CHOICE).unwrap();
        let mut e = c.evaluators.remove(0);
        e.sample = 0.5;
        let mut other = e.clone();
        other.name = "other".into();
        let ids: Vec<_> = (0..1000u128)
            .map(|i| (i << 64 | 0x7000).to_be_bytes())
            .collect();
        let same = ids
            .iter()
            .filter(|id| e.sampled(id.as_slice()) == other.sampled(id.as_slice()))
            .count();
        assert!((400..=600).contains(&same), "{same}");
    }
}
