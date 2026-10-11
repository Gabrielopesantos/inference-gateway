# inference-gateway

A gateway between LLM harnesses (Claude Code, Codex, OpenAI-SDK clients) and model providers. It forwards
traffic unchanged and observes every exchange:

- tokens, cost, TTFB, TTFT, decode throughput, errors and rate-limit headroom as Prometheus metrics
- one OpenTelemetry span per request, with `gen_ai.*` attributes
- every request as a row in SQLite, and prompt/response content when a capture rule matches
- asynchronous quality scores for captured responses from a System One judge model (tested:
  [Typesafe Jev](https://docs.typesafe.ai))

Grafana shows all of it; there is no custom UI.

Requests are passed through in the provider's native format (Anthropic Messages, OpenAI Chat Completions), so
prompt caching, thinking and beta headers work as without the gateway. Client API keys pass through; the gateway
holds no provider key.

## Quick start

Needs Nix and Docker (or Podman).

```sh
nix build .#image && docker load < result
docker compose -f deploy/compose.yaml up -d
```

Then send traffic through `http://localhost:8080` and open Grafana at http://localhost:3000. The
[deploy README](deploy/README.md) covers the stack, ports, secrets and evaluation.

Without containers:

```sh
nix develop
cargo run -- --config config/gateway.example.toml
```

## How it works

```
client ──► ingress ──► proxy ──► upstream
                         │ copy of the response stream
                         ▼
                    observation ──► one record per request
                         │
                       cost (sets cost_usd)
                         │
       ┌─────────┬───────┴────┬──────────┐
   access log  prometheus   store      otel        one bounded queue each
                              │
                          eval jobs ──► judge ──► scores
```

- The proxy forwards the request and streams the response back. It never waits on anything below it.
- A copy of the response feeds the provider adapter's parser, which yields tokens, timings, stop reason and
  errors. Every request ends in exactly one record: completed, failed or cancelled by the client.
- Each exporter has its own worker behind a bounded queue. A slow exporter drops records, counted in
  `gateway_module_dropped_total`. It does not slow the proxy or the other exporters.
- The store writes the trace row, the captured content and the evaluation jobs in one SQLite transaction. Workers
  score the jobs later, in the gateway or in a separate `eval run` process.

## Harness setup

Clients reach an upstream at `/{upstream name}/...`.

| harness | setting |
|---|---|
| Claude Code | `ANTHROPIC_BASE_URL=http://localhost:8080/anthropic` |
| OpenAI SDK, Codex | `OPENAI_BASE_URL=http://localhost:8080/openai/v1` |

Optional request headers, read and then stripped before forwarding:

| header | effect |
|---|---|
| `x-gateway-session: <id>` | groups the calls of one agent run; Claude Code's `x-claude-code-session-id` is used when absent |
| `x-gateway-tag-<key>: <value>` | free-form tags, matched by capture rules and evaluators |
| `x-gateway-capture: 0\|1` | per-request capture, if the matching rule sets `allow_header_override` |
| `x-gateway-token` | ingress token, when `[ingress] require_token = true` |

Claude Code sends extra headers with `ANTHROPIC_CUSTOM_HEADERS="x-gateway-tag-project: demo"`. A W3C
`traceparent` header is forwarded and puts the gateway span into the caller's trace.

## Configuration

Two sources, each setting in exactly one:

- **Flags and `IG_*` env vars** for process settings (listen addresses, TLS, timeouts, log format):
  `inference-gateway --help`. Secrets are env only: `IG_IDENTITY_SECRET` (HMAC key for caller ids),
  `IG_GATEWAY_TOKEN`, and the judge key named in `[eval.judge] api_key_env`.
- **A TOML file** for upstreams, ingress, capture rules and modules: see the commented
  [`config/gateway.example.toml`](config/gateway.example.toml). A module (`[cost]`, `[prometheus]`, `[store]`,
  `[eval]`, `[otel]`) is on when its section is present.

Evaluation also runs standalone against the same database: `inference-gateway eval run --config <file>`, and
`inference-gateway eval backfill --evaluator NAME` scores already captured traces.

## Capture and privacy

- Content is stored only for requests that match a capture rule; the default is no capture.
- **With capture on, the database holds prompts and responses.** Keep it on an encrypted volume, keep rules narrow
  and keep the retention TTL (`[capture] retention`, default 30 days; trace rows without content are kept).
- API keys are never logged, stored or exported. Caller ids are an HMAC of the key, so they are stable but not
  reversible.
- Spans and metrics carry no content.
- A gateway reachable from other machines relays for anyone who can reach it: set `[ingress] allow_cidrs` and/or
  `require_token`.

## Cost

`[cost]` sets `cost_usd` per request from [LiteLLM's public price list](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).
The gateway fetches the list at startup and refreshes it every 6 hours. It caches the list on disk for the next
start. The cost is an estimate:

- Standard-tier prices, including the higher rates some models charge for long prompts. Batch, flex and priority
  prices are not read.
- A model the list lacks gets no cost and one warning in the log. Price it under `[cost.models]`; the list wins
  once it has the model.
- A request keeps the cost computed when it finished. Later price changes do not rewrite stored rows.

## Limits

- Only Anthropic Messages and OpenAI Chat Completions are observed. Other paths pass through and get a trace
  row with status and duration, but no tokens, cost or content. This includes the OpenAI Responses API, which
  current Codex versions use.
- The request body is buffered, up to `--max-body` (default 32 MiB). The response is streamed.
- The gateway asks upstreams for uncompressed responses (`accept-encoding: identity`), so it can parse the copy.
- Trace rows are never deleted. Only captured content has a retention.
- One SQLite file with one writer: one gateway process per database.

## Metrics

Served on the admin listener (`--admin-listen`, default `127.0.0.1:9090`) at `/metrics` when `[prometheus]` is set.
Labels: `upstream, provider, model` (plus `outcome, streaming` on requests and duration).

| metric | what |
|---|---|
| `gateway_requests_total` | requests by outcome: `ok`, `upstream_error`, `client_cancelled`, `gateway_error` |
| `gateway_tokens_total{kind}` | `input`, `output`, `cache_read`, `cache_write`, `reasoning` |
| `gateway_cost_usd_total` | estimated spend |
| `gateway_ttfb_seconds`, `gateway_ttft_seconds` | first byte, first content delta (thinking included) |
| `gateway_request_duration_seconds` | until the last byte |
| `gateway_output_tokens_per_second` | output tokens / (end − first token) |
| `gateway_chunk_gap_seconds` | mean gap between response chunks |
| `gateway_upstream_ratelimit_remaining{kind}` | from provider rate-limit headers (API-key traffic) |
| `gateway_subscription_window_utilization{window}` | 5h and 7d windows (subscription traffic) |
| `gateway_module_dropped_total{module}` | records an export module dropped because its queue was full |
| `gateway_eval_jobs{status}`, `gateway_eval_scores_total`, `gateway_eval_judge_duration_seconds`, `gateway_eval_confidence`, `gateway_eval_failures_total` | evaluation worker |

## Traces

`[otel]` sends one span per request as OTLP/HTTP JSON to `{endpoint}/v1/traces`. The trace id is the request id
(the `traces.id` row in SQLite), or the caller's trace id when it sent a `traceparent` (then stored in
`traces.otel_trace_id`). Attributes: `gen_ai.provider.name`, `gen_ai.operation.name`, `gen_ai.request.model`,
`gen_ai.response.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
`gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_creation.input_tokens`,
`gen_ai.response.finish_reasons`, HTTP method, path and status, `session.id`, and `gateway.*` timings, cost,
caller and tags. Events `first_byte` and `first_token`.

## Token normalization

Providers count input differently. The gateway reports:

- `input` = every prompt token, cache reads and writes included. Anthropic's `input_tokens` excludes cache tokens,
  so they are added; OpenAI's `prompt_tokens` already includes cached tokens.
- `cache_read`, `cache_write` (Anthropic only; `cache_write_1h` is the part with the 1-hour TTL) and `reasoning`
  (OpenAI `reasoning_tokens`, Anthropic thinking tokens) separately.

For streaming OpenAI requests the gateway adds `stream_options.include_usage` when absent, so usage is observed.
Clients then receive a final chunk with empty `choices`; `inject_stream_usage = false` on the upstream turns it
off.

## License

[MIT](LICENSE)
