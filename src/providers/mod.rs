//! Provider kinds. Adapters that parse provider traffic arrive in phase 2.

use std::fmt;

use serde::Deserialize;

/// Wire format of an upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    Anthropic,
    Openai,
}

impl fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Anthropic => "anthropic",
            Self::Openai => "openai",
        })
    }
}
