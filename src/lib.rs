//! `inference-gateway` - a transparent streaming gateway for LLM APIs.

pub mod capture;
pub mod config;
pub mod error;
#[cfg(feature = "eval")]
pub mod eval;
pub mod gateway;
pub mod identity;
pub mod ingress;
pub mod modules;
pub mod observe;
pub mod providers;
pub mod proxy;
pub mod redact;
pub mod server;
#[cfg(feature = "store")]
pub mod store;
pub mod trace_context;
