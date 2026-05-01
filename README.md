# rinha-2026

Participação na [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

## Subir localmente

```
docker compose up --build
```

A API fica disponível em `http://localhost:9999`.

## Endpoints

- `GET /ready` — retorna 200 quando pronto, 503 durante startup
- `POST /fraud-score` — classifica transação e retorna score de fraude

## Topologia

Produção: 2 instâncias API comunicam com o nginx via Unix sockets (`/sockets/api1.sock`, `/sockets/api2.sock`). Sem TCP local entre LB e APIs.

Dev (sem Docker): `LISTEN_TCP=0.0.0.0:8080 cargo run --release`

## Testes

```
cargo test
```

Testes unitários cobrindo scorer direto, busca k-NN quantizada, dataset e respostas HTTP.

## Benchmark

```
cargo bench
```

Mede scorer direto, busca quantizada e pipeline end-to-end. Meta: `score_body` < 100µs em hardware local.

## Teste de carga (k6 oficial)

Pré-requisito: [k6](https://grafana.com/docs/k6/latest/set-up/install-k6/) instalado no host.

```
./run-test.sh
```

Sobe a stack, aguarda `/ready`, roda o k6 oficial contra `:9999`, exibe `results.json` com `final_score`. Grava uso de CPU/memória por container em `docker-stats.log` durante o teste. Derruba a stack ao final.
