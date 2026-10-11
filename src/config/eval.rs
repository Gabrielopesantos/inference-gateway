//! `[eval]`: the judge and the evaluators that score captured traces.

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
    #[serde(default = "default_concurrency")]
    pub concurrency: usize,
    #[serde(default = "default_max_attempts")]
    pub max_attempts: u32,
    #[serde(default = "default_lease", with = "humantime_serde")]
    pub lease: Duration,
    #[serde(default = "default_poll", with = "humantime_serde")]
    pub poll: Duration,
    pub judge: JudgeFile,
    #[serde(default)]
    pub evaluators: Vec<EvaluatorFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JudgeFile {
    pub base_url: String,
    pub model: String,
    pub api_key_env: String,
    #[serde(default = "default_max_state_tokens")]
    pub max_state_tokens: usize,
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
    pub state: String,
    pub question: Question,
    pub pass: Option<PassFile>,
    pub min_confidence: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum PassFile {
    /// noul: `yes` or `no`.
    Answer(String),
    /// choice and score: labels from `criteria`.
    Labels(Vec<String>),
}

/// Sent to the judge as is; also hashed into the evaluator version.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum Question {
    /// Probability of yes.
    Noul {
        instructions: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    /// One label out of `criteria`.
    Choice {
        instructions: String,
        /// Option → description; `""` is sent as `null` (TOML has no null).
        #[serde(serialize_with = "empty_as_null")]
        criteria: BTreeMap<String, String>,
    },
    /// Level index into `criteria`, lowest first.
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NoulCriteria {
    #[serde(rename = "true", default, skip_serializing_if = "Option::is_none")]
    pub yes: Option<String>,
    #[serde(rename = "false", default, skip_serializing_if = "Option::is_none")]
    pub no: Option<String>,
}

fn empty_as_null<S: Serializer>(map: &BTreeMap<String, String>, s: S) -> Result<S::Ok, S::Error> {
    s.collect_map(map.iter().map(|(k, v)| (k, (!v.is_empty()).then_some(v))))
}

impl Question {
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Noul { .. } => "noul",
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

/// Jev limit: state plus the longest question.
fn default_max_state_tokens() -> usize {
    32_000
}

fn default_timeout() -> Duration {
    Duration::from_mins(1)
}

/// Jev list price; output tokens are not billed.
fn default_usd_per_mtok_input() -> f64 {
    0.042
}

#[derive(Debug, Clone)]
pub struct EvalConfig {
    /// Max jobs in flight, which is also max concurrent judge calls.
    pub concurrency: usize,
    pub max_attempts: u32,
    /// How long a claimed job is held before another worker may take it.
    pub lease: Duration,
    /// Queue poll interval; new jobs from this process wake the worker sooner.
    pub poll: Duration,
    pub judge: Judge,
    pub evaluators: Vec<Evaluator>,
}

#[derive(Debug, Clone)]
pub struct Judge {
    pub base_url: Url,
    /// Pin a version: an alias such as `jev-latest` changes scores under
    /// the same evaluator version.
    pub model: String,
    /// Env var holding the key; read when the worker starts.
    pub api_key_env: String,
    pub max_state_tokens: usize,
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
    pub state: String,
    pub question: Question,
    /// Score labels that pass, as stored (noul: `true` / `false`); `None`
    /// gives no verdict. Not part of the version.
    pub pass: Option<Vec<String>>,
    /// Below it the verdict is `uncertain`; every score is stored.
    pub min_confidence: Option<f64>,
    /// Changes with the judge model, state template or question.
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
    if file.concurrency == 0 {
        bail!("eval.concurrency must be at least 1");
    }
    if file.max_attempts == 0 {
        bail!("eval.max_attempts must be at least 1");
    }
    let j = file.judge;
    let judge = Judge {
        base_url: parse_base_url(&j.base_url).context("eval.judge.base_url")?,
        model: j.model,
        api_key_env: j.api_key_env,
        max_state_tokens: j.max_state_tokens,
        timeout: j.timeout,
        usd_per_mtok_input: j.usd_per_mtok_input,
    };
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
        concurrency: file.concurrency,
        max_attempts: file.max_attempts,
        lease: file.lease,
        poll: file.poll,
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
    match &e.question {
        Question::Noul { .. } => {}
        Question::Choice { criteria, .. } => {
            if !(1..=255).contains(&criteria.len()) {
                bail!("choice needs 1 to 255 criteria, got {}", criteria.len());
            }
        }
        Question::Score { criteria, .. } => {
            if !(2..=10).contains(&criteria.len()) {
                bail!("score needs 2 to 10 criteria, got {}", criteria.len());
            }
        }
    }
    minijinja::Environment::new()
        .template_from_str(&e.state)
        .context("state template")?;
    let pass = e
        .pass
        .map(|p| pass_labels(p, &e.question))
        .transpose()
        .context("pass")?;
    let version = version(model, &e.state, &e.question)?;
    Ok(Evaluator {
        name: e.name,
        matcher,
        sample: e.sample,
        state: e.state,
        question: e.question,
        pass,
        min_confidence: e.min_confidence,
        version,
    })
}

fn pass_labels(pass: PassFile, question: &Question) -> Result<Vec<String>> {
    let labels = match (question, pass) {
        (Question::Noul { .. }, PassFile::Answer(a)) => match a.as_str() {
            "yes" => return Ok(vec!["true".into()]),
            "no" => return Ok(vec!["false".into()]),
            _ => bail!("noul takes \"yes\" or \"no\", got {a:?}"),
        },
        (Question::Noul { .. }, PassFile::Labels(_)) => {
            bail!("noul takes \"yes\" or \"no\", not a list")
        }
        (q, PassFile::Answer(_)) => bail!("{} takes a list of criteria", q.kind()),
        (_, PassFile::Labels(l)) => l,
    };
    if labels.is_empty() {
        bail!("empty list");
    }
    for l in &labels {
        let known = match question {
            Question::Choice { criteria, .. } => criteria.contains_key(l),
            Question::Score { criteria, .. } => criteria.contains(l),
            Question::Noul { .. } => false,
        };
        if !known {
            bail!("{l:?} is not in criteria");
        }
    }
    Ok(labels)
}

fn version(model: &str, state: &str, question: &Question) -> Result<String> {
    let mut h = Sha256::new();
    for part in [model, state, &serde_json::to_string(question)?] {
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
state = "<user>{{ last_user_message }}</user>"

[eval.evaluators.question]
type = "choice"
instructions = "Did it answer?"

[eval.evaluators.question.criteria]
yes = "Fully"
no = ""
"#;

    fn parse(extra: &str) -> Result<EvalConfig> {
        Config::parse(&format!("{BASE}{extra}")).map(|c| c.eval.unwrap())
    }

    fn err(extra: &str) -> String {
        format!("{:#}", parse(extra).unwrap_err())
    }

    #[test]
    fn parses_with_defaults() {
        let c = parse(CHOICE).unwrap();
        assert_eq!(c.concurrency, 4);
        assert_eq!(c.max_attempts, 5);
        assert_eq!(c.lease, Duration::from_mins(5));
        assert_eq!(c.judge.max_state_tokens, 32_000);
        let e = &c.evaluators[0];
        assert_eq!(e.question.kind(), "choice");
        assert!((e.sample - 1.0).abs() < f64::EPSILON);
        assert_eq!(e.version.len(), 16);
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
        let q: Question =
            toml::from_str("type = \"noul\"\ninstructions = \"x\"\ncriteria = { true = \"y\" }\n")
                .unwrap();
        assert_eq!(
            serde_json::to_value(&q).unwrap(),
            serde_json::json!({"type": "noul", "instructions": "x", "criteria": {"true": "y"}})
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

        let score = |levels: &str| {
            err(&format!(
                "[[eval.evaluators]]\nname = \"s\"\nstate = \"x\"\n\n[eval.evaluators.question]\ntype = \"score\"\ninstructions = \"i\"\ncriteria = {levels}\n"
            ))
        };
        assert!(score("[\"one\"]").contains("score needs 2 to 10"));
        assert!(score(&format!("{:?}", vec!["l"; 11])).contains("score needs 2 to 10"));

        let e = err(
            "[[eval.evaluators]]\nname = \"c\"\nstate = \"x\"\nquestion = { type = \"choice\", instructions = \"i\", criteria = {} }\n",
        );
        assert!(
            e.contains(r#"eval.evaluators[0] "c": choice needs 1 to 255"#),
            "{e}"
        );

        let e = err(&CHOICE.replace("{{ last_user_message }}", "{{ unclosed"));
        assert!(e.contains("state template"), "{e}");

        let e = err(&CHOICE.replace("name = \"answered\"", "name = \"a b\""));
        assert!(e.contains("name"), "{e}");

        let e = err(&CHOICE.replace(
            "name = \"answered\"\n",
            "name = \"answered\"\nmatch = { upstream = \"nope\" }\n",
        ));
        assert!(e.contains("match.upstream"), "{e}");
    }

    #[test]
    fn pass_maps_to_stored_labels() {
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
            ("[\"maybe\"]", r#""maybe" is not in criteria"#),
            ("[]", "empty list"),
            ("\"yes\"", "choice takes a list"),
        ] {
            let e = format!("{:#}", with(pass).unwrap_err());
            assert!(e.contains(&format!("pass: {msg}")), "{pass}: {e}");
        }

        let noul = |pass: &str| {
            parse(&format!(
                "[[eval.evaluators]]\nname = \"n\"\nstate = \"x\"\npass = {pass}\nquestion = {{ type = \"noul\", instructions = \"i\" }}\n"
            ))
        };
        assert_eq!(
            noul("\"no\"").unwrap().evaluators[0].pass,
            Some(vec!["false".to_owned()])
        );
        assert!(format!("{:#}", noul("\"true\"").unwrap_err()).contains("noul takes"));
        assert!(format!("{:#}", noul("[\"yes\"]").unwrap_err()).contains("not a list"));

        let score = parse(
            "[[eval.evaluators]]\nname = \"s\"\nstate = \"x\"\npass = [\"good\"]\nquestion = { type = \"score\", instructions = \"i\", criteria = [\"bad\", \"good\"] }\n",
        )
        .unwrap();
        assert_eq!(score.evaluators[0].pass, Some(vec!["good".to_owned()]));
    }

    #[test]
    fn unknown_question_type_reports_line() {
        let e = err(&CHOICE.replace("type = \"choice\"", "type = \"rank\""));
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
