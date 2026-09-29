"""mitmproxy addon: save Anthropic /v1/messages responses as fixture candidates.

    mitmdump --mode reverse:https://api.anthropic.com@8082 -s scripts/record_fixtures.py
    ANTHROPIC_BASE_URL=http://localhost:8082 claude   # in a throwaway directory

Writes $RECORD_DIR (default recorded/) <n>-<model>-<kinds>.{sse,json,headers,request}.
`.request` holds the request headers (credentials dropped) and the body's
`metadata`, to find a stable session id; the rest of the request body is never
saved. Responses hold the assistant's output: review before copying into
tests/fixtures/anthropic/.
"""

import json
import os
import re

from mitmproxy import http

OUT = os.environ.get("RECORD_DIR", "recorded")
DROP = re.compile(
    r"^(authorization|x-api-key|set-cookie|cookie|cf-|anthropic-organization-id|anthropic-workspace-id|"
    r"strict-transport-security|alt-svc|server|via|x-envoy|date|"
    r"content-encoding|content-length|transfer-encoding)",
    re.IGNORECASE,
)


def kinds(body: bytes, status: int) -> list[str]:
    found = []
    if b"thinking_delta" in body or b'"type":"thinking"' in body:
        found.append("thinking")
    if b"input_json_delta" in body or b'"type":"tool_use"' in body:
        found.append("tool_use")
    if re.search(rb'"cache_read_input_tokens":\s*[1-9]', body):
        found.append("cache_hit")
    if re.search(rb'"cache_creation_input_tokens":\s*[1-9]', body):
        found.append("cache_write")
    if status >= 400:
        found.append(f"http{status}")
    return found or ["text"]


def response(flow: http.HTTPFlow) -> None:
    req, resp = flow.request, flow.response
    if resp is None or req.method != "POST" or not req.path.split("?")[0].endswith("/v1/messages"):
        return
    body = resp.content or b""  # decoded, even if the client asked for gzip
    try:
        req_json = json.loads(req.content or b"{}")
    except ValueError:
        req_json = {}
    if not isinstance(req_json, dict):
        req_json = {}
    model = str(req_json.get("model", "unknown"))
    model = re.sub(r"[^A-Za-z0-9._-]", "_", model)

    os.makedirs(OUT, exist_ok=True)
    n = sum(1 for f in os.listdir(OUT) if f.endswith(".headers")) + 1
    stem = os.path.join(OUT, f"{n:03d}-{model}-{'+'.join(kinds(body, resp.status_code))}")
    ext = ".sse" if resp.headers.get("content-type", "").startswith("text/event-stream") else ".json"
    with open(stem + ext, "wb") as f:
        f.write(body)
    with open(stem + ".headers", "w") as f:
        f.write(f"{resp.http_version.replace('.0', '')} {resp.status_code}\n")
        for name, value in resp.headers.items(multi=True):
            if not DROP.match(name):
                f.write(f"{name}: {value}\n")
    with open(stem + ".request", "w") as f:
        for name, value in req.headers.items(multi=True):
            if not DROP.match(name):
                f.write(f"{name}: {value}\n")
        f.write(f"\nmetadata: {json.dumps(req_json.get('metadata'))}\n")
