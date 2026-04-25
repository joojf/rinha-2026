use bytes::Bytes;
use http::{response::Builder, StatusCode};
use monoio_http::{
    common::response::Response,
    h1::payload::{FixedPayload, Payload},
};

const BODIES: [&[u8]; 6] = [
    br#"{"approved":true,"fraud_score":0.0}"#,
    br#"{"approved":true,"fraud_score":0.2}"#,
    br#"{"approved":true,"fraud_score":0.4}"#,
    br#"{"approved":false,"fraud_score":0.6}"#,
    br#"{"approved":false,"fraud_score":0.8}"#,
    br#"{"approved":false,"fraud_score":1.0}"#,
];

pub fn ok_fraud_score(fraud_count: u8) -> Response {
    let body = BODIES[fraud_count.min(5) as usize];
    Builder::new()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .header("content-length", body.len().to_string())
        .body(Payload::Fixed(FixedPayload::new(Bytes::from_static(body))))
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
