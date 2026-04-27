use bytes::Bytes;
use http::Method;
use monoio::io::stream::Stream;
use monoio_http::{common::response::Response, h1::payload::Payload};
use std::sync::atomic::Ordering;

use crate::{payload, response, search, vectorize, DATASET, MCC, NORM, READY};

pub async fn handle_request(req: http::Request<Payload>) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_owned();

    match (method, path.as_str()) {
        (Method::GET, "/ready") => {
            drain_body(req.into_body()).await;
            if READY.load(Ordering::Acquire) {
                response::ok_ready()
            } else {
                response::not_ready()
            }
        }
        (Method::POST, "/fraud-score") => handle_fraud_score(req.into_body()).await,
        _ => {
            drain_body(req.into_body()).await;
            response::not_found()
        }
    }
}

async fn handle_fraud_score(body: Payload) -> Response {
    let bytes = match read_body(body).await {
        Some(b) => b,
        None => return response::ok_fraud_score(0),
    };

    let req = match payload::parse(&bytes) {
        Ok(r) => r,
        Err(_) => return response::ok_fraud_score(0),
    };

    let norm = NORM.get().unwrap();
    let mcc = MCC.get().unwrap();
    let ds = DATASET.get().unwrap();

    let q = vectorize::vectorize(&req, norm, mcc);
    let fraud_count = search::knn5_fraud_count_ivf(&q, ds);
    response::ok_fraud_score(fraud_count)
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
