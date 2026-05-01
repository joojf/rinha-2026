use monoio::{
    io::{AsyncReadRent, AsyncWriteRent, AsyncWriteRentExt},
    net::{TcpListener, UnixListener},
};
use std::sync::atomic::Ordering;

use crate::{DATASET, READY, scorer};

const READ_CHUNK: usize = 4096;
const MAX_PENDING: usize = 16384;

const READY_OK: &[u8] =
    b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";
const READY_NOT_YET: &[u8] =
    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";
const NOT_FOUND: &[u8] =
    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";

const FRAUD_RESPONSES: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];

pub async fn accept_loop_tcp(listener: TcpListener) {
    loop {
        if let Ok((stream, _)) = listener.accept().await {
            monoio::spawn(handle_connection(stream));
        }
    }
}

pub async fn accept_loop_uds(listener: UnixListener) {
    loop {
        if let Ok((stream, _)) = listener.accept().await {
            monoio::spawn(handle_connection(stream));
        }
    }
}

async fn handle_connection<T>(mut stream: T)
where
    T: AsyncReadRent + AsyncWriteRent + 'static,
{
    let mut pending = Vec::with_capacity(READ_CHUNK * 2);
    let mut read_buf = Vec::with_capacity(READ_CHUNK);

    loop {
        read_buf.clear();
        let (res, buf) = stream.read(read_buf).await;
        read_buf = buf;

        let n = match res {
            Ok(0) | Err(_) => return,
            Ok(n) => n,
        };

        pending.extend_from_slice(&read_buf[..n]);
        if pending.len() > MAX_PENDING {
            return;
        }

        while let Some((consumed, resp)) = response_for_pending(&pending) {
            let (res, returned_resp) = stream.write_all(resp).await;
            let _ = returned_resp;
            if res.is_err() {
                return;
            }

            if consumed == pending.len() {
                pending.clear();
            } else {
                pending.drain(..consumed);
            }
        }
    }
}

pub fn response_for_http_request(req: &[u8]) -> Option<&'static [u8]> {
    let (_, resp) = response_for_pending(req)?;
    Some(resp)
}

fn response_for_pending(buf: &[u8]) -> Option<(usize, &'static [u8])> {
    let header_end = find_header_end(buf)?;
    let header_len = header_end + 4;

    if buf.starts_with(b"GET /ready ") {
        return Some((
            header_len,
            if READY.load(Ordering::Acquire) {
                READY_OK
            } else {
                READY_NOT_YET
            },
        ));
    }

    let content_len = content_length(&buf[..header_end]).unwrap_or(0);
    let total_len = header_len.checked_add(content_len)?;
    if buf.len() < total_len {
        return None;
    }

    if buf.starts_with(b"POST /fraud-score ") {
        let body = &buf[header_len..total_len];
        let fraud_count = DATASET
            .get()
            .and_then(|ds| scorer::score_body(body, ds).ok())
            .unwrap_or(0);
        return Some((total_len, fraud_response(fraud_count)));
    }

    Some((total_len, NOT_FOUND))
}

#[inline]
fn fraud_response(fraud_count: u8) -> &'static [u8] {
    FRAUD_RESPONSES[fraud_count.min(5) as usize]
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    let mut i = 0usize;
    while i + 3 < buf.len() {
        if &buf[i..i + 4] == b"\r\n\r\n" {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn content_length(headers: &[u8]) -> Option<usize> {
    const NAME: &[u8] = b"content-length:";
    let last = headers.len().saturating_sub(NAME.len());
    let mut i = 0usize;
    while i <= last {
        if (i == 0 || headers[i - 1] == b'\n')
            && eq_ignore_ascii_case(&headers[i..i + NAME.len()], NAME)
        {
            return parse_usize(skip_ws(&headers[i + NAME.len()..]));
        }
        i += 1;
    }
    None
}

#[inline]
fn skip_ws(mut s: &[u8]) -> &[u8] {
    while matches!(s.first(), Some(b' ' | b'\t')) {
        s = &s[1..];
    }
    s
}

fn parse_usize(s: &[u8]) -> Option<usize> {
    let mut n = 0usize;
    let mut seen = false;
    for &b in s {
        match b {
            b'0'..=b'9' => {
                seen = true;
                n = n.checked_mul(10)?.checked_add((b - b'0') as usize)?;
            }
            b' ' | b'\t' | b'\r' | b'\n' => break,
            _ => return None,
        }
    }
    seen.then_some(n)
}

fn eq_ignore_ascii_case(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.eq_ignore_ascii_case(y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::READY;

    #[test]
    fn ready_response_switches_by_atomic() {
        READY.store(false, Ordering::Release);
        assert_eq!(
            response_for_http_request(b"GET /ready HTTP/1.1\r\nHost: x\r\n\r\n").unwrap(),
            READY_NOT_YET
        );
        READY.store(true, Ordering::Release);
        assert_eq!(
            response_for_http_request(b"GET /ready HTTP/1.1\r\nHost: x\r\n\r\n").unwrap(),
            READY_OK
        );
    }

    #[test]
    fn waits_for_complete_body() {
        assert!(
            response_for_http_request(
                b"POST /fraud-score HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc",
            )
            .is_none()
        );
    }

    #[test]
    fn invalid_fraud_body_returns_score_zero() {
        let req = b"POST /fraud-score HTTP/1.1\r\nContent-Length: 3\r\n\r\nabc";
        let (consumed, resp) = response_for_pending(req).unwrap();
        assert_eq!(consumed, req.len());
        assert!(std::str::from_utf8(resp).unwrap().contains("\"fraud_score\":0.0"));
    }

    #[test]
    fn not_found_consumes_declared_body() {
        let (consumed, resp) =
            response_for_pending(b"POST /x HTTP/1.1\r\nContent-Length: 3\r\n\r\nabcGET").unwrap();
        assert_eq!(consumed, 42);
        assert_eq!(resp, NOT_FOUND);
    }

    #[test]
    fn response_lengths_match_body_lengths() {
        for resp in FRAUD_RESPONSES {
            let text = std::str::from_utf8(resp).unwrap();
            let body = text.split("\r\n\r\n").nth(1).unwrap();
            let content_len = text
                .lines()
                .find_map(|line| line.strip_prefix("Content-Length: "))
                .unwrap()
                .parse::<usize>()
                .unwrap();
            assert_eq!(body.len(), content_len);
        }
    }
}
