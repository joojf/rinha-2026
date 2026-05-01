use criterion::{Criterion, criterion_group, criterion_main};
use rinha_2026::{
    DATASET, dataset::Dataset, scorer, server, search::knn5_fraud_count_ivf_i16,
};
use std::hint::black_box;

static SAMPLE: &[u8] = br#"{
    "id": "tx-bench-001",
    "transaction": { "amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z" },
    "customer": { "avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009", "MERC-001"] },
    "merchant": { "id": "MERC-001", "mcc": "5912", "avg_amount": 298.95 },
    "terminal": { "is_online": false, "card_present": true, "km_from_home": 13.709 },
    "last_transaction": { "timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.862 }
}"#;

fn bench_vectorize_body(c: &mut Criterion) {
    c.bench_function("vectorize_body_i16", |b| {
        b.iter(|| scorer::vectorize_body_i16(black_box(SAMPLE)).unwrap())
    });
}

fn bench_search(c: &mut Criterion) {
    let ds = Dataset::load_embedded().unwrap();
    let q = [
        3000i16, 5000, 1000, 7000, 2000, 4000, 6000, 500, 9000, 10000, 0, 0, 5000, 1000,
    ];

    c.bench_function("knn5_ivf_i16", |b| {
        b.iter(|| knn5_fraud_count_ivf_i16(black_box(&q), &ds))
    });
}

fn bench_end_to_end(c: &mut Criterion) {
    let ds = Dataset::load_embedded().unwrap();

    c.bench_function("score_body", |b| {
        b.iter(|| scorer::score_body(black_box(SAMPLE), &ds).unwrap())
    });
}

fn bench_raw_http_endpoint(c: &mut Criterion) {
    let _ = DATASET.set(Dataset::load_embedded().unwrap());
    let mut req = Vec::with_capacity(SAMPLE.len() + 96);
    req.extend_from_slice(
        b"POST /fraud-score HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: ",
    );
    req.extend_from_slice(SAMPLE.len().to_string().as_bytes());
    req.extend_from_slice(b"\r\n\r\n");
    req.extend_from_slice(SAMPLE);

    c.bench_function("raw_http_endpoint", |b| {
        b.iter(|| server::response_for_http_request(black_box(&req)).unwrap())
    });
}

criterion_group!(
    benches,
    bench_vectorize_body,
    bench_search,
    bench_end_to_end,
    bench_raw_http_endpoint
);
criterion_main!(benches);
