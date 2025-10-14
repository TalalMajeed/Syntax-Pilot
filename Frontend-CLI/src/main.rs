use std::env;
use std::io::{self, Write};
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

    // Send query to local AI server
    let client = reqwest::Client::new();
    let res = client
        .post("http://localhost:8000/query")
        .json(&QueryRequest { query })
        .send()
        .await?;

    if !res.status().is_success() {
        eprintln!("Error: server returned status {}", res.status());
        std::process::exit(1);
    }

    let body: QueryResponse = res.json().await?;
    let command = body.response.trim();

    // Confirm execution
    print!("Confirm: {}? ", command);
    io::stdout().flush()?;
    let mut confirm = String::new();
    io::stdin().read_line(&mut confirm)?;
    if !confirm.trim().is_empty() {
        println!("Command rejected.");
        return Ok(());
    }

    // Execute the command
    let status = std::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .status()?;

    if status.success() {
        println!("Command executed successfully.");
    } else {
        println!("Command failed with status: {:?}", status);
    }

    Ok(())
}