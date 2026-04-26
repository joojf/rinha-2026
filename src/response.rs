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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bodies_exact_content() {
        assert_eq!(BODIES[0], br#"{"approved":true,"fraud_score":0.0}"#);
        assert_eq!(BODIES[1], br#"{"approved":true,"fraud_score":0.2}"#);
        assert_eq!(BODIES[2], br#"{"approved":true,"fraud_score":0.4}"#);
        assert_eq!(BODIES[3], br#"{"approved":false,"fraud_score":0.6}"#);
        assert_eq!(BODIES[4], br#"{"approved":false,"fraud_score":0.8}"#);
        assert_eq!(BODIES[5], br#"{"approved":false,"fraud_score":1.0}"#);
    }

    #[test]
    fn score_boundary_approved_vs_rejected() {
        for c in 0u8..3 {
            assert!(
                std::str::from_utf8(BODIES[c as usize]).unwrap().contains("true"),
                "fraud_count={c} deve ser approved"
            );
        }
        for c in 3u8..=5 {
            assert!(
                std::str::from_utf8(BODIES[c as usize]).unwrap().contains("false"),
                "fraud_count={c} deve ser rejected"
            );
        }
    }

    #[test]
    fn score_overflow_clamps_to_5() {
        assert_eq!(BODIES[(99u8).min(5) as usize], BODIES[5]);
    }

    #[test]
    fn ok_fraud_score_status_and_headers() {
        let r = ok_fraud_score(0);
        assert_eq!(r.status(), StatusCode::OK);
        assert_eq!(r.headers().get("content-type").unwrap(), "application/json");
    }

    #[test]
    fn ready_status_200() {
        assert_eq!(ok_ready().status(), StatusCode::OK);
    }

    #[test]
    fn not_ready_status_503() {
        assert_eq!(not_ready().status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
