use mimalloc::MiMalloc;
use std::sync::{OnceLock, atomic::AtomicBool};

pub mod dataset;
pub mod handler;
pub mod mcc_risk;
pub mod normalization;
pub mod payload;
pub mod response;
pub mod search;
pub mod server;
pub mod vectorize;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub static READY: AtomicBool = AtomicBool::new(false);
pub static NORM: OnceLock<normalization::Normalization> = OnceLock::new();
pub static MCC: OnceLock<mcc_risk::MccRisk> = OnceLock::new();
pub static DATASET: OnceLock<dataset::Dataset> = OnceLock::new();
