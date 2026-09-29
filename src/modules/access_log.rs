//! One `tracing` line per request; format follows `--log-format`.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::info;

use super::{CompletedExchange, Module, Phase};

pub struct AccessLog;

#[async_trait]
impl Module for AccessLog {
    fn name(&self) -> &'static str {
        "access_log"
    }

    fn phase(&self) -> Phase {
        Phase::Export
    }

    async fn export(&self, ex: Arc<CompletedExchange>) {
        let r = &ex.record;
        info!(
            target: "access_log",
            id = %r.id,
            upstream = %r.upstream,
            method = %r.method,
            path = %r.path,
            model = r.response_model.as_deref().or(r.requested_model.as_deref()),
            outcome = r.outcome.as_str(),
            status = r.http_status,
            error_type = r.error_type.as_deref(),
            caller = r.principal.caller_id.as_deref(),
            session = r.principal.session_id.as_deref(),
            input_tokens = r.usage.input_tokens_total,
            output_tokens = r.usage.output_tokens,
            cost_usd = r.cost_usd,
            ttft_ms = r.timings.ttft.map(|d| d.as_millis()),
            duration_ms = r.timings.duration.as_millis(),
            "request finished"
        );
    }
}
