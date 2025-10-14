use std::env;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct QueryRequest {
    query: String,
}

#[derive(Deserialize)]
struct QueryResponse {
    response: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: ecello-pilot <english command description>");
        std::process::exit(1);
    }

    let query = args.join(" ");

    let client = reqwest::Client::new();
    let res = client
        .post("http://localhost:8000/query")
        .json(&QueryRequest { query })
        .send()
        .await;

    let res = match res {
        Ok(r) => r,
        Err(e) => {
            if e.is_connect() {
                eprintln!("Connection Error");
                std::process::exit(1);
            } else {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    };

    if !res.status().is_success() {
        eprintln!("Error: server returned status {}", res.status());
        std::process::exit(1);
    }

    let body: QueryResponse = match res.json().await {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Error: failed to parse server response");
            std::process::exit(1);
        }
    };

    // ✅ Just print the suggested command
    println!("{}", body.response.trim());

    Ok(())
}