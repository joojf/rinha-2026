use std::os::unix::fs::PermissionsExt;
use std::sync::atomic::Ordering;
use monoio::{net::TcpListener, IoUringDriver};

use rinha_2026::{
    dataset, mcc_risk, normalization,
    search, server,
    DATASET, MCC, NORM, READY,
};

fn main() {
    NORM.set(normalization::Normalization::load_embedded()).ok();
    MCC.set(mcc_risk::MccRisk::load_embedded()).ok();
    DATASET.set(dataset::Dataset::load_embedded().expect("falha ao carregar dataset")).ok();

    let uds_path = std::env::var("LISTEN_UDS").ok();
    let tcp_addr = std::env::var("LISTEN_TCP").unwrap_or_else(|_| "0.0.0.0:8080".to_string());

    monoio::start::<IoUringDriver, _>(async move {
        warm_up();
        READY.store(true, Ordering::Release);

        if let Some(path) = uds_path {
            std::fs::remove_file(&path).ok();
            let opts = monoio::net::ListenerOpts::new()
                .reuse_port(false)
                .reuse_addr(false);
            let listener = monoio::net::UnixListener::bind_with_config(&path, &opts)
                .expect("unix bind failed");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666))
                .expect("chmod socket failed");
            server::accept_loop_uds(listener).await;
        } else {
            let listener = TcpListener::bind(&tcp_addr).expect("tcp bind failed");
            server::accept_loop_tcp(listener).await;
        }
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
