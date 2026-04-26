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
