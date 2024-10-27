use dashmap::DashMap;
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{ AsyncReadExt, AsyncWriteExt };
use std::net::SocketAddr;
use anyhow::{anyhow, Result as ARes};
use std::sync::Arc;
use crate::common;

#[derive(Clone)]
pub struct Server {
    store: Arc<DashMap<Vec<u8>, Vec<u8>>>,
    addr: SocketAddr,
}

impl Server {
    pub fn new(full_addr: String) -> Self {
        Server {
            store: Arc::new(DashMap::new()),
            addr: full_addr.parse().unwrap(),
        }
    }

    // start the server and consume the main thread
    pub async fn start(&mut self) -> ARes<()> {
        let listener = TcpListener::bind(self.addr).await?;
        println!("Server listening on {}", self.addr);

        // wait for incoming client connections
        loop {
            let (mut sock, addr) = listener.accept().await?;
            println!("Server received connection from {}", addr);

            let mut server_clone = self.clone();
            // spawn a new thread for each client
            tokio::spawn(async move {
                while let Ok(()) = server_clone.dispatch(&mut sock).await {}
                println!("Client {} disconnected or an error occurred.", addr);
            });
        }
    }

    // dispatch request to the respective function
    async fn dispatch(&mut self, sock: &mut TcpStream) -> ARes<()> {
        // read header to determine appropriate function
        let mut buf = [0; common::HEADER_LEN];
        let res = match sock.read(&mut buf).await {
            Ok(n) => {
                match &buf[..n] {
                    b"PUT" => self.recv_put(sock).await,
                    b"DEL" => self.recv_del(sock).await,
                    b"GET" => self.recv_get(sock).await,
                    _ => Err(anyhow!("Invalid header received"))
                }
            }
            Err(e) => Err(anyhow!("Failed to read from socket: {}", e)),
        };

        match res {
            Ok(_) => {}, // OK is sent by the dispatch functions because they are not all the same
            Err(_) => self.send_err(sock).await?,
        }

        Ok(())
    }

    // receive a put request
    async fn recv_put(&mut self, sock: &mut TcpStream) -> ARes<()> {
        // read the key length and key
        let key_len = self.recv_len(sock).await?;
        let mut key = vec![0; key_len];
        sock.read_exact(&mut key).await?;
        // read val length and val
        let val_len = self.recv_len(sock).await?;
        let mut val = vec![0; val_len];
        sock.read_exact(&mut val).await?;
        // insert key/val into store
        self.store.insert(key, val);
        // send ok
        self.send_ok(sock).await?;
        Ok(())
    }

    // receive a del request
    async fn recv_del(&mut self, sock: &mut TcpStream) -> ARes<()> {
        // read key len and key
        let key_len = self.recv_len(sock).await?;
        let mut key = vec![0; key_len];
        sock.read_exact(&mut key).await?;
        // try to delete key. send ok if key exists, err otheriwse
        match self.store.remove(&key) {
            Some(_) => self.send_ok(sock).await?,
            None => self.send_err(sock).await?,
        }
        Ok(())
    }

    // receive a get request
    async fn recv_get(&mut self, sock: &mut TcpStream) -> ARes<()> {
        // read key len and key
        let key_len = self.recv_len(sock).await?;
        let mut key = vec![0; key_len];
        sock.read_exact(&mut key).await?;
        // try to get key. send ok if key exists, err otheriwse
        let val = match self.store.get(&key) {
            Some(v) => v.clone(),
            None => return Err(anyhow!("Key not in store")),
        };
        // write val to socket
        let mut buf = Vec::new();
        let mut status_buf = [0; common::STATUS_LEN];
        let message = b"OK";
        status_buf[..message.len()].copy_from_slice(message);
        buf.extend_from_slice(&status_buf);
        buf.extend_from_slice(&val.len().to_be_bytes());
        buf.extend_from_slice(&val);
        sock.write_all(&buf).await?;
        Ok(())
    }

    // get length from request
    async fn recv_len(&mut self, sock: &mut TcpStream) -> ARes<usize> {
        let mut buf = [0; std::mem::size_of::<usize>()];
        sock.read_exact(&mut buf).await?;
        let len = usize::from_be_bytes(buf);
        Ok(len)
    }

    // send ok to client
    async fn send_ok(&mut self, sock: &mut TcpStream) -> ARes<()> {
        let mut buf = [0; common::STATUS_LEN];
        let message = b"OK";
        buf[..message.len()].copy_from_slice(message);

        sock.write_all(&buf).await?;
        Ok(())
    }

    // send error to client
    async fn send_err(&mut self, sock: &mut TcpStream) -> ARes<()> {
        let mut buf = [0; common::STATUS_LEN];
        let message = b"ERROR";
        buf[..message.len()].copy_from_slice(message);

        sock.write_all(&buf).await?;
        Ok(())
    }
}
