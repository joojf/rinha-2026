use bytes::Bytes;
use http::{
    HeaderValue, StatusCode,
    header::{CONTENT_LENGTH, CONTENT_TYPE},
};
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
const BODY_LENS: [&str; 6] = ["35", "35", "35", "36", "36", "36"];

pub fn ok_fraud_score(fraud_count: u8) -> Response {
    let idx = fraud_count.min(5) as usize;
    let body = BODIES[idx];
    let mut resp = Response::new(Payload::Fixed(FixedPayload::new(Bytes::from_static(body))));
    *resp.status_mut() = StatusCode::OK;
    resp.headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    resp.headers_mut()
        .insert(CONTENT_LENGTH, HeaderValue::from_static(BODY_LENS[idx]));
    resp
}

pub fn ok_ready() -> Response {
    empty(StatusCode::OK)
}

pub fn not_ready() -> Response {
    empty(StatusCode::SERVICE_UNAVAILABLE)
}

pub fn not_found() -> Response {
    empty(StatusCode::NOT_FOUND)
}

fn empty(status: StatusCode) -> Response {
    let mut resp = Response::new(Payload::None);
    *resp.status_mut() = status;
    resp.headers_mut()
        .insert(CONTENT_LENGTH, HeaderValue::from_static("0"));
    resp
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
        for i in 0..BODIES.len() {
            assert_eq!(BODY_LENS[i], BODIES[i].len().to_string());
        }
    }

    #[test]
    fn score_boundary_approved_vs_rejected() {
        for c in 0u8..3 {
            assert!(
                std::str::from_utf8(BODIES[c as usize])
                    .unwrap()
                    .contains("true"),
                "fraud_count={c} deve ser approved"
            );
        }
        for c in 3u8..=5 {
            assert!(
                std::str::from_utf8(BODIES[c as usize])
                    .unwrap()
                    .contains("false"),
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
