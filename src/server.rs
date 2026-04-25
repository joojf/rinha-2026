use monoio::{
    io::{sink::SinkExt, stream::Stream, Splitable},
    net::{TcpListener, TcpStream},
};
use monoio_http::{
    common::{error::HttpError, request::Request},
    h1::codec::{decoder::RequestDecoder, encoder::GenericEncoder},
    util::spsc::{spsc_pair, SPSCReceiver},
};

use crate::handler::handle_request;

pub async fn accept_loop(listener: TcpListener) {
    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                monoio::spawn(handle_connection(stream));
            }
            Err(_) => {}
        }
    }
}

async fn handle_connection(stream: TcpStream) {
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
