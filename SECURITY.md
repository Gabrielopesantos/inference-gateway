# Security

Report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/Gabrielopesantos/inference-gateway/security/advisories/new),
not in a public issue.

In scope:

- client API keys or gateway secrets leaking into logs, the store, spans, metrics or error responses
- bypass of `[ingress]` (`allow_cidrs`, `require_token`)
- requests reaching an upstream other than the configured `base_url` and its path
- captured content readable by someone who should not read it, or kept past its retention
- a crash or unbounded resource use caused by a client or an upstream

The deploy stack in `deploy/` is for local use: its Grafana has anonymous Admin access by design.
