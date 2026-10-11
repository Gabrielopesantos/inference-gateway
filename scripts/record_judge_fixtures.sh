#!/usr/bin/env bash
# Record System One judge responses (Typesafe Jev API) as
# tests/fixtures/judge/<name>.{json,headers}.
#
#     IG_JUDGE_API_KEY=... scripts/record_judge_fixtures.sh
#
# The questions equal the ones in tests/eval_judge.rs, with question id `q`.
# The state is synthetic. Credentials and cookies are dropped from headers.
set -euo pipefail

: "${IG_JUDGE_API_KEY:?set IG_JUDGE_API_KEY}"
URL="${IG_JUDGE_BASE_URL:-https://api.typesafe.ai}/v1/systemone"
MODEL="${IG_JUDGE_MODEL:-jev-1.13.0}"
OUT="$(dirname "$0")/../tests/fixtures/judge"
STATE='<user>hi there</user><response>Hello! How can I help you today?</response>'
DROP='^(authorization|x-api-key|set-cookie|cookie|cf-|strict-transport-security|alt-svc|server|via|x-envoy|date|content-encoding|content-length|transfer-encoding)'

mkdir -p "$OUT"

# record NAME KEY QUESTION_JSON
record() {
    local name=$1 key=$2 question=$3 raw
    raw=$(mktemp)
    jq -n --arg model "$MODEL" --arg state "$STATE" --argjson q "$question" \
        '{model: $model, state: $state, questions: {q: $q}}' |
        curl -sS -X POST "$URL" \
            -H "Authorization: Bearer $key" \
            -H 'Content-Type: application/json' \
            --data-binary @- -D "$raw" -o "$OUT/$name.json"
    tr -d '\r' <"$raw" | grep -viE "$DROP" | sed '/^$/d' >"$OUT/$name.headers"
    rm -f "$raw"
    echo "$name: $(head -n 1 "$OUT/$name.headers")"
}

record noul "$IG_JUDGE_API_KEY" \
    '{"type": "noul", "instructions": "Is the response polite?"}'
record choice "$IG_JUDGE_API_KEY" \
    '{"type": "choice", "instructions": "Did the response address the request?",
      "criteria": {"correct": "Fully answers", "partial": "Answers part", "wrong": null}}'
record score "$IG_JUDGE_API_KEY" \
    '{"type": "score", "instructions": "How helpful is the response?",
      "criteria": ["bad", "ok", "great"]}'
record error_401 "not-a-key" \
    '{"type": "noul", "instructions": "Is the response polite?"}'
record error_422 "$IG_JUDGE_API_KEY" \
    '{"type": "choice", "instructions": "Did the response address the request?"}'
