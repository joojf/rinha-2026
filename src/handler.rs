use bytes::Bytes;
use http::Method;
use monoio::io::stream::Stream;
use monoio_http::{common::response::Response, h1::payload::Payload};
use std::sync::atomic::Ordering;

use crate::{DATASET, READY, response, scorer};

pub async fn handle_request(req: http::Request<Payload>) -> Response {
    if req.method() == Method::GET && req.uri().path() == "/ready" {
        drain_body(req.into_body()).await;
        if READY.load(Ordering::Acquire) {
            response::ok_ready()
        } else {
            response::not_ready()
        }
    } else if req.method() == Method::POST && req.uri().path() == "/fraud-score" {
        handle_fraud_score(req.into_body()).await
    } else {
        drain_body(req.into_body()).await;
        response::not_found()
    }
}

async fn handle_fraud_score(body: Payload) -> Response {
    let bytes = match read_body(body).await {
        Some(b) => b,
        None => return response::ok_fraud_score(0),
    };

    let ds = DATASET.get().unwrap();

    match scorer::score_body(&bytes, ds) {
        Ok(fraud_count) => response::ok_fraud_score(fraud_count),
        Err(_) => response::ok_fraud_score(0),
    }
}

async fn read_body(body: Payload) -> Option<Bytes> {
    if let Payload::Fixed(mut fp) = body {
        if let Some(Ok(data)) = fp.next().await {
            return Some(data);
        }
    }
    None
}

async fn drain_body(body: Payload) {
    if let Payload::Fixed(mut fp) = body {
        let _ = fp.next().await;
    }
}
