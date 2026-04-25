use mimalloc::MiMalloc;
use monoio::{net::TcpListener, IoUringDriver};
use std::sync::{OnceLock, atomic::{AtomicBool, Ordering}};

mod dataset;
mod handler;
mod mcc_risk;
mod normalization;
mod payload;
mod response;
mod search;
mod server;
mod vectorize;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub static READY: AtomicBool = AtomicBool::new(false);
pub static NORM: OnceLock<normalization::Normalization> = OnceLock::new();
pub static MCC: OnceLock<mcc_risk::MccRisk> = OnceLock::new();
pub static DATASET: OnceLock<dataset::Dataset> = OnceLock::new();

fn main() {
    NORM.set(normalization::Normalization::load_embedded()).ok();
    MCC.set(mcc_risk::MccRisk::load_embedded()).ok();
    DATASET.set(dataset::Dataset::load_embedded().expect("falha ao carregar dataset")).ok();

    monoio::start::<IoUringDriver, _>(async {
        warm_up();
        READY.store(true, Ordering::Release);

        let listener = TcpListener::bind("0.0.0.0:8080").expect("bind failed");
        server::accept_loop(listener).await;
    });
}

fn warm_up() {
    use std::time::Instant;
    let ds = DATASET.get().unwrap();
    search::warm_up_buffer(ds.padded_len);

    let start = Instant::now();
    let mut state = 0x12345678u32;
    for _ in 0..1000 {
        let mut q = [0.0f32; 14];
        for v in q.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (state >> 8) as f32 / (1u32 << 24) as f32;
        }
        let _ = search::knn5_fraud_count(&q, ds);
    }
    let elapsed = start.elapsed();
    eprintln!("warm-up: {:?} (1000 buscas)", elapsed);
    debug_assert!(elapsed.as_millis() < 500);
}
