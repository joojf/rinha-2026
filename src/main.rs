use mimalloc::MiMalloc;
use monoio::{net::TcpListener, IoUringDriver};
use std::sync::atomic::{AtomicBool, Ordering};

mod handler;
mod payload;
mod response;
mod server;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

pub static READY: AtomicBool = AtomicBool::new(false);

fn main() {
    READY.store(true, Ordering::Release);

    monoio::start::<IoUringDriver, _>(async {
        let listener = TcpListener::bind("0.0.0.0:8080").expect("bind failed");
        server::accept_loop(listener).await;
    });
}
