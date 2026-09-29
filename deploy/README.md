# Local metrics

Prometheus and Grafana for a gateway running on the host. Requires the
`[prometheus]` section in the gateway config.

```sh
cargo run -- --config config/gateway.example.toml
docker compose -f deploy/compose.yaml up -d
```

- Grafana: http://localhost:3000 (no login; "Gateway" dashboard is home)
- Prometheus: http://localhost:9091 (port 9090 is the gateway admin listener)

`docker compose -f deploy/compose.yaml down -v` stops the stack and deletes
stored samples.

Host networking works on Linux only. Elsewhere, drop `network_mode: host`,
publish ports 9091 and 3000, and set the scrape target to
`host.docker.internal:9090` and the datasource URL to `http://prometheus:9091`.
