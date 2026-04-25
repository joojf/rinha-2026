use mimalloc::MiMalloc;
use monoio::{net::TcpListener, IoUringDriver};
use std::sync::{OnceLock, atomic::{AtomicBool, Ordering}};

mod dataset;
mod handler;
mod mcc_risk;
mod normalization;
mod payload;
mod response;
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
    READY.store(true, Ordering::Release);

    monoio::start::<IoUringDriver, _>(async {
        let listener = TcpListener::bind("0.0.0.0:8080").expect("bind failed");
        server::accept_loop(listener).await;
    });
}
