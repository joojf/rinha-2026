# rinha-2026 — submission

Branch da submissão pra [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

Código-fonte na branch `main`: https://github.com/joojf/rinha-2026/tree/main

## Subir

```
docker compose up -d
```

API responde em `http://localhost:9999/fraud-score` (POST) e `/ready` (GET).

## Topologia

- 2 instâncias da API (`ghcr.io/joojf/rinha-2026:latest`) em monoio + io_uring
- nginx 1.27 como LB, comunica com APIs via Unix sockets
- Limites somam ≤ 1.0 CPU e ≤ 350 MB
- `cpu_period=10ms` por container reduz throttle worst-case do cgroup pra 5.5ms

## Stack

Rust nightly + monoio (runtime io_uring single-thread) + nginx + AVX2/FMA pra busca k-NN exata.
