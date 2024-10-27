use std::thread;
use std::time::Duration;
use tokio::runtime;
use assert_ok::assert_ok;
use crate::{ client::Client, server::Server };

static SERVER_ADDR: &str = "127.0.0.1:1895";

// unit tests for broad functionality
// I would not use these in production

#[tokio::test]
async fn test_connect() {
    start_server().await;
    let mut client = Client::new(SERVER_ADDR.to_string());
    assert_ok!(client.connect().await);
}

#[tokio::test]
async fn test_put() {
    start_server().await;
    let mut client = Client::new(SERVER_ADDR.to_string());
    assert_ok!(client.connect().await);
    assert_ok!(client.put("hello".as_bytes(), "world".as_bytes()).await);
}

#[tokio::test]
async fn test_get() {
    start_server().await;
    let mut client = Client::new(SERVER_ADDR.to_string());
    assert_ok!(client.connect().await);
    assert_ok!(client.put("hello".as_bytes(), "world".as_bytes()).await);
    let gres = client.get("hello".as_bytes()).await;
    assert_eq!(gres.unwrap().val, Some("world".as_bytes().to_vec()));
}

#[tokio::test]
async fn test_del() {
    start_server().await;
    let mut client = Client::new(SERVER_ADDR.to_string());
    assert_ok!(client.connect().await);
    assert_ok!(client.put("hello".as_bytes(), "world".as_bytes()).await);
    let gres = client.del("hello".as_bytes()).await;
    assert_eq!(gres.unwrap().val, None);
}

// start the server on another thread to prevent blocking
async fn start_server() {
    thread::spawn(move || {
        let rt = runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mut server = Server::new(SERVER_ADDR.to_string());
            let _ = server.start().await;
        });
    });
    // give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;
}
