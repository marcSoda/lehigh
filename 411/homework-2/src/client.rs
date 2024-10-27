use tokio::net::TcpStream;
use tokio::io::{ AsyncReadExt, AsyncWriteExt };
use std::net::SocketAddr;
use anyhow::Result as ARes;
use std::str;
pub use crate::common::{ Resp, Status, STATUS_LEN };

pub struct Client {
    addr: SocketAddr,
    sock: Option<TcpStream>,
}

impl Client {
    pub fn new(full_addr: String) -> Self {
        Client {
            addr: full_addr.parse().unwrap(),
            sock: None,
        }
    }

    // connect client to server
    pub async fn connect(&mut self) -> ARes<()> {
        self.sock = Some(TcpStream::connect(self.addr).await?);
        println!("Client connected to {}", self.addr);
        Ok(())
    }

    // send put request
    pub async fn put(&mut self, key: &[u8], val: &[u8]) -> ARes<Resp> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"PUT");
        buf.extend_from_slice(&key.len().to_be_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&val.len().to_be_bytes());
        buf.extend_from_slice(val);
        self.sock.as_mut().unwrap().write_all(&buf).await?;
        let status = self.recv_status().await?;
        Ok(Resp {
            status,
            val: None,
        })
    }

    // send del request
    pub async fn del(&mut self, key: &[u8]) -> ARes<Resp> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"DEL");
        buf.extend_from_slice(&key.len().to_be_bytes());
        buf.extend_from_slice(key);
        self.sock.as_mut().unwrap().write_all(&buf).await?;
        let status = self.recv_status().await?;
        Ok(Resp {
            status,
            val: None,
        })
    }

    // send get request
    pub async fn get(&mut self, key: &[u8]) -> ARes<Resp> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GET");
        buf.extend_from_slice(&key.len().to_be_bytes());
        buf.extend_from_slice(key);
        self.sock.as_mut().unwrap().write_all(&buf).await?;
        let status = self.recv_status().await?;

        if status == Status::ERR {
            return Ok(Resp {
                status,
                val: None,
            })
        }

        let len = self.recv_len().await?;
        let mut val = vec![0; len];
        self.sock.as_mut().unwrap().read_exact(&mut val).await?;

        Ok(Resp {
            status,
            val: Some(val),
        })
    }

    // get status from server after sending request
    pub async fn recv_status(&mut self) -> ARes<Status> {
        let mut buf = [0; STATUS_LEN];
        self.sock.as_mut().unwrap().read_exact(&mut buf).await?;

        let status = if let Ok(s) = str::from_utf8(&buf) {
            if s.contains("OK") { Status::OK }
            else if s.contains("ERROR") { Status::ERR }
            else { Status::UNKNOWN }
        } else { Status::UNKNOWN };

        Ok(status)
    }

    // get the val length from get response
    async fn recv_len(&mut self) -> ARes<usize> {
        let mut buf = [0; std::mem::size_of::<usize>()];
        self.sock.as_mut().unwrap().read_exact(&mut buf).await?;
        let len = usize::from_be_bytes(buf);
        Ok(len)
    }
}
