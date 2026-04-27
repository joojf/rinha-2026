use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use rinha_2026::{
    dataset::Dataset,
    mcc_risk::MccRisk,
    normalization::Normalization,
    payload,
    search::{knn5_fraud_count_blocks, knn5_fraud_count_ivf},
    vectorize::vectorize,
};

static SAMPLE: &[u8] = br#"{
    "id": "tx-bench-001",
    "transaction": { "amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z" },
    "customer": { "avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009", "MERC-001"] },
    "merchant": { "id": "MERC-001", "mcc": "5912", "avg_amount": 298.95 },
    "terminal": { "is_online": false, "card_present": true, "km_from_home": 13.709 },
    "last_transaction": { "timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.862 }
}"#;

fn bench_parse(c: &mut Criterion) {
    c.bench_function("payload_parse", |b| {
        b.iter(|| payload::parse(black_box(SAMPLE)).unwrap())
    });
}

fn bench_vectorize(c: &mut Criterion) {
    let norm = Normalization::load_embedded();
    let mcc = MccRisk::load_embedded();
    let req = payload::parse(SAMPLE).unwrap();

    c.bench_function("vectorize", |b| {
        b.iter(|| vectorize(black_box(&req), &norm, &mcc))
    });
}

fn bench_search(c: &mut Criterion) {
    let ds = Dataset::load_embedded().unwrap();
    let q = [0.3f32, 0.5, 0.1, 0.7, 0.2, 0.4, 0.6, 0.05, 0.9, 1.0, 0.0, 0.0, 0.5, 0.1];

    c.bench_function("knn5_ivf", |b| {
        b.iter(|| knn5_fraud_count_ivf(black_box(&q), &ds))
    });

    c.bench_function("knn5_blocks", |b| {
        b.iter(|| knn5_fraud_count_blocks(black_box(&q), &ds))
    });
}

fn bench_end_to_end(c: &mut Criterion) {
    let norm = Normalization::load_embedded();
    let mcc = MccRisk::load_embedded();
    let ds = Dataset::load_embedded().unwrap();

    c.bench_function("score_end_to_end", |b| {
        b.iter(|| {
            let req = payload::parse(black_box(SAMPLE)).unwrap();
            let q = vectorize(&req, &norm, &mcc);
            knn5_fraud_count_ivf(&q, &ds)
        })
    });
}

criterion_group!(benches, bench_parse, bench_vectorize, bench_search, bench_end_to_end);
criterion_main!(benches);
