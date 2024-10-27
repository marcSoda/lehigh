use h2::server::Server;
use h2::client::Client;
use std::env;
use rand::Rng;
use rand::SeedableRng;
use anyhow::{Error as AErr, Result as ARes};
use std::time::Instant;

// launch a server or run a client benchmark
#[tokio::main]
async fn main() -> ARes<()> {
    // parse args
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input_string>", args[0]);
        std::process::exit(1);
    }
    let typ = &args[1];

    match typ.as_str() {
        "c" => {
            // parse client args
            if args.len() != 6 {
                eprintln!("clients, ops, max_k, ratio");
                std::process::exit(1);
            }
            let num_clients: usize = args[2].parse().unwrap(); // number of clients to spawn
            let num_ops: usize = args[3].parse().unwrap();     // number of operations per client
            let max_k: usize = args[4].parse().unwrap();       // max key value
            let ratio: usize = args[5].parse().unwrap();       // read-only ratio

            let start_time = Instant::now();
            let mut tasks = vec![];

            // for each client
            for _ in 0..num_clients {
                let task = tokio::spawn(async move {
                    // spawn client
                    let mut client = Client::new("104.131.105.204:1895".to_string());
                    client.connect().await.map_err(AErr::from)?;
                    let mut rng = rand::rngs::StdRng::from_entropy();
                    // operate
                    run_operations(&mut rng, client, num_ops, max_k, ratio).await?;

                    Ok::<(), AErr>(())
                });
                tasks.push(task);
            }

            for task in tasks {
                let _ = task.await.unwrap();
            }
            // get throughput
            let elapsed_time = start_time.elapsed();
            let total_operations = num_clients * num_ops;
            let throughput = total_operations as f64 / elapsed_time.as_secs_f64();
            println!("Throughput: {} ops/sec", throughput);
        },
        "s" => {
            // launch server
            let mut server = Server::new("0.0.0.0:1895".to_string());
            server.start().await.map_err(AErr::from)?;
        },
        _ => println!("bad args"),
    }
    Ok(())
}

// execute i operations
async fn run_operations(rng: &mut rand::rngs::StdRng, mut client: Client, i: usize, k: usize, r: usize) -> ARes<()> {
    // for every operation
    for _ in 0..i {
        // get random value
        let val = (rng.gen_range(0..k) + i).to_string();
        // use read-only ratio to choose an operation
        let op_type: usize = rng.gen_range(0..100);
        if op_type < r {
            client.get(val.as_bytes()).await?;
        } else if op_type < (100 + r) / 2 {
            client.put(val.as_bytes(), val.as_bytes()).await?;
        } else {
            client.del(val.as_bytes()).await?;
        }
    }
    Ok(())
}
