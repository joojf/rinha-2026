#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

cleanup() {
    kill "$STATS_PID" 2>/dev/null || true
    docker compose down -v --remove-orphans 2>/dev/null || true
}
STATS_PID=0
trap cleanup EXIT

echo "==> docker compose up --build -d"
docker compose up --build -d

echo "==> aguardando /ready"
for i in $(seq 1 60); do
    if curl -fsS http://localhost:9999/ready >/dev/null 2>&1; then
        echo "    ready em ${i}s"
        break
    fi
    sleep 1
    if [[ $i -eq 60 ]]; then
        echo "    timeout: /ready nao respondeu em 60s"
        docker compose logs
        exit 1
    fi
done

STATS_LOG="$(pwd)/docker-stats.log"
: > "$STATS_LOG"
(
    while true; do
        docker stats --no-stream --format \
            'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' \
            >> "$STATS_LOG"
        echo "---" >> "$STATS_LOG"
        sleep 2
    done
) &
STATS_PID=$!

echo "==> rodando k6 oficial"
pushd spec >/dev/null
k6 run test/test.js
popd >/dev/null

echo ""
echo "==> resultado:"
jq '.' spec/test/results.json

echo ""
echo "==> final_score: $(jq '.scoring.final_score' spec/test/results.json)"
echo "==> docker stats salvo em docker-stats.log"
