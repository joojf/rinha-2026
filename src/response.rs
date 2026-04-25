use bytes::Bytes;
use http::{response::Builder, StatusCode};
use monoio_http::{
    common::response::Response,
    h1::payload::{FixedPayload, Payload},
};

const BODY_FRAUD_OK: &[u8] = b"{\"approved\":true,\"fraud_score\":0.0}";

pub fn ok_fraud_score() -> Response {
    Builder::new()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .header("content-length", BODY_FRAUD_OK.len().to_string())
        .body(Payload::Fixed(FixedPayload::new(Bytes::from_static(
            BODY_FRAUD_OK,
        ))))
        .unwrap()
}

pub fn ok_ready() -> Response {
    Builder::new()
        .status(StatusCode::OK)
        .header("content-length", "0")
        .body(Payload::None)
        .unwrap()
}

pub fn not_ready() -> Response {
    Builder::new()
        .status(StatusCode::SERVICE_UNAVAILABLE)
        .header("content-length", "0")
        .body(Payload::None)
        .unwrap()
}

pub fn not_found() -> Response {
    Builder::new()
        .status(StatusCode::NOT_FOUND)
        .header("content-length", "0")
        .body(Payload::None)
        .unwrap()
}
