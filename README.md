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
