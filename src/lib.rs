use mimalloc::MiMalloc;
use std::sync::{OnceLock, atomic::AtomicBool};

pub mod dataset;
pub mod handler;
pub mod response;
pub mod scorer;
pub mod search;
pub mod server;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub static READY: AtomicBool = AtomicBool::new(false);
pub static DATASET: OnceLock<dataset::Dataset> = OnceLock::new();
