use monoio::{
    io::{AsyncReadRent, AsyncWriteRent, Split, Splitable, sink::SinkExt, stream::Stream},
    net::{TcpListener, UnixListener},
};
use monoio_http::{
    common::{error::HttpError, request::Request},
    h1::codec::{decoder::RequestDecoder, encoder::GenericEncoder},
    util::spsc::{SPSCReceiver, spsc_pair},
};

use crate::handler::handle_request;

pub async fn accept_loop_tcp(listener: TcpListener) {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                monoio::spawn(handle_connection(stream));
            }
            Err(_) => {}
        }
    }
}

pub async fn accept_loop_uds(listener: UnixListener) {
    loop {
        match listener.accept().await {
            Ok((stream, _)) => {
                monoio::spawn(handle_connection(stream));
            }
            Err(_) => {}
        }
    }
}

async fn handle_connection<T>(stream: T)
where
    T: Split + AsyncReadRent + AsyncWriteRent + 'static,
{
    let (r, w) = stream.into_split();
    let encoder = GenericEncoder::new(w);
    let mut decoder = RequestDecoder::new(r);
    let (mut tx, rx) = spsc_pair();

    monoio::spawn(handle_task(rx, encoder));

    loop {
        match decoder.next().await {
            Some(Ok(req)) => {
                if tx.send(req).await.is_err() {
                    return;
                }
            }
            _ => return,
        }
    }
}

async fn handle_task(
    mut rx: SPSCReceiver<Request>,
    mut encoder: impl monoio::io::sink::Sink<
        monoio_http::common::response::Response,
        Error = impl Into<HttpError>,
    >,
) {
    loop {
        let req = match rx.recv().await {
            Some(r) => r,
            None => return,
        };
        let resp = handle_request(req).await;
        if encoder.send_and_flush(resp).await.is_err() {
            return;
        }
    }
}
