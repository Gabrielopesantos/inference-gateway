# syntax=docker/dockerfile:1
# docker build -t inference-gateway:dev .
ARG RUST_VERSION=1.95

FROM docker.io/library/rust:${RUST_VERSION}-slim-bookworm AS build
WORKDIR /src
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked \
    && cp target/release/inference-gateway /inference-gateway
RUN mkdir /data

FROM gcr.io/distroless/cc-debian12
COPY --from=build /inference-gateway /usr/local/bin/inference-gateway
# Writable store directory for `nobody`, the same user as the Nix image.
COPY --from=build --chown=65534:65534 /data /data
USER 65534:65534
WORKDIR /data
ENV IG_LISTEN=0.0.0.0:8080 \
    IG_ADMIN_LISTEN=0.0.0.0:9090 \
    IG_CONFIG=/etc/inference-gateway/gateway.toml \
    IG_LOG_FORMAT=json
EXPOSE 8080 9090
VOLUME /data
ENTRYPOINT ["inference-gateway"]
