# Observability stack

The gateway plus [`grafana/otel-lgtm`](https://github.com/grafana/docker-otel-lgtm)
(Grafana, Prometheus, Tempo and an OpenTelemetry collector in one container).

```sh
docker compose -f deploy/compose.yaml up -d --build
```

Compose builds the image from the repository's `Dockerfile`. With Nix,
`nix build .#image && docker load < result` builds the same image.

- Gateway: http://localhost:8080 (`GATEWAY_PORT=8088` publishes another port)
- Grafana: http://localhost:3000 (no login; "Gateway" dashboard is home, "Quality" next to it)
- Prometheus: http://localhost:9091
- OTLP/HTTP: http://localhost:4318

Every port is published on 127.0.0.1 only. Keep it so: Grafana runs with
anonymous Admin access and reads captured prompts through its SQLite datasource,
and `gateway.toml` allows the private address ranges the published port arrives
from.

Point a harness at the gateway:

```sh
ANTHROPIC_BASE_URL=http://localhost:8080/anthropic \
ANTHROPIC_CUSTOM_HEADERS="x-gateway-tag-project: demo" claude
OPENAI_BASE_URL=http://localhost:8080/openai/v1 codex
```

The `project: demo` tag matches the capture rule in `gateway.toml`, so those
requests are stored with content and can be scored.

## Files

| file | what |
|---|---|
| `gateway.toml` | gateway config inside the container: store on the `gateway-data` volume, spans to `lgtm:4318` |
| `.env` (optional, gitignored) | `IG_IDENTITY_SECRET` (caller ids), `IG_JUDGE_API_KEY` (evaluation), `IG_GATEWAY_TOKEN`; copy `.env.example` |
| `prometheus.yaml` | otel-lgtm's default plus scrape jobs for the gateway |
| `grafana/provisioning/datasources/datasources.yaml` | Prometheus, Tempo, Loki, Pyroscope (otel-lgtm's uids) and the two gateway SQLite stores |
| `grafana/dashboards/*.json` | provisioned dashboards |

Evaluation is off until the `[eval.*]` sections in `gateway.toml` are uncommented
and `IG_JUDGE_API_KEY` is set in `deploy/.env`. After editing either, run
`docker compose -f deploy/compose.yaml up -d --force-recreate gateway`.

## Gateway on the host

`docker compose -f deploy/compose.yaml up -d lgtm` starts only the backend. Run
the gateway with a copy of the example config (`config/gateway.local.toml` is
gitignored) that sets `[otel] endpoint = "http://localhost:4318"`, and with
`--admin-listen 0.0.0.0:9090`: the `gateway-host` scrape job reaches it through
`host.docker.internal`, which a loopback-only listener does not accept. That
exposes the unauthenticated `/metrics` to the network too; firewall the port, or
listen on the Docker bridge address instead. The
`store` dropdown on the dashboards picks the SQLite store: `Gateway SQLite`
(the `gateway-data` volume) or `Gateway SQLite (host)` (`config/gateway.db`,
mounted read-only). The host store can be read only while its `-wal` and `-shm`
files exist, that is while the host gateway runs.

## SQLite from Grafana

Grafana opens the store read-only (`mode=ro`, plus the plugin's
`query_only` pragma). The volume is mounted read-write on purpose: the store
uses WAL mode, and a WAL reader must write the `-shm` file next to the database.

`docker compose -f deploy/compose.yaml down -v` stops the stack and deletes the
store, metrics and traces.
