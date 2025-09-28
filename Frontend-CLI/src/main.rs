use anyhow::{Result, anyhow};
use dotenv::dotenv;
use std::{env, sync::Arc};
use ort::{Environment, SessionBuilder, Value, Session};
use tokenizers::Tokenizer;
use ndarray::{Array3, ArrayD, IxDyn, CowArray};
use serde::Deserialize;
use serde_json::json;
use reqwest::Client;

/// Pinecone match structure
#[derive(Deserialize, Debug)]
struct Match {
    score: f32,
    metadata: Option<serde_json::Value>,
}

#[derive(Deserialize, Debug)]
struct QueryResponse {
    matches: Vec<Match>,
}

/// Mean pooling for sentence embeddings
fn mean_pooling(token_embeddings: Array3<f32>, attention_mask: Vec<i64>) -> Vec<f32> {
    let hidden_size = token_embeddings.shape()[2];
    let seq_len = token_embeddings.shape()[1];

    let mut pooled = vec![0f32; hidden_size];
    let mut count = 0f32;

    for (i, &mask) in attention_mask.iter().enumerate().take(seq_len) {
        if mask == 1 {
            for j in 0..hidden_size {
                pooled[j] += token_embeddings[[0, i, j]];
            }
            count += 1.0;
        }
    }

    pooled.iter_mut().for_each(|v| *v /= count.max(1.0));
    pooled
}

/// Run embedding with ONNX model
fn embed_text(session: &Session, tokenizer: &Tokenizer, text: &str) -> Result<Vec<f32>> {
    let encoding = tokenizer.encode(text, true).map_err(|e| anyhow!(e))?;
    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();

    let input_ids = CowArray::from(ArrayD::from_shape_vec(IxDyn(&[1, input_ids.len()]), input_ids)?);
    let attention_mask_arr = CowArray::from(ArrayD::from_shape_vec(IxDyn(&[1, attention_mask.len()]), attention_mask.clone())?);
    let token_type_ids = CowArray::from(ArrayD::from_shape_vec(IxDyn(&[1, input_ids.len()]), vec![0i64; input_ids.len()])?);

    let inputs = vec![
        Value::from_array(session.allocator(), &input_ids)?,
        Value::from_array(session.allocator(), &attention_mask_arr)?,
        Value::from_array(session.allocator(), &token_type_ids)?,
    ];

    let outputs = session.run(inputs)?;
    let tensor = outputs[0].try_extract()?;
    let tensor_view = tensor.view();
    let dims = tensor_view.shape();
    let data: Vec<f32> = tensor_view.iter().copied().collect();
    let token_embeddings = Array3::from_shape_vec((dims[0], dims[1], dims[2]), data)?;
    Ok(mean_pooling(token_embeddings, attention_mask))
}

/// Query Pinecone with embedding
async fn query_pinecone(client: &Client, api_key: &str, index_url: &str, vector: Vec<f32>) -> Result<Vec<Match>> {
    let body = json!({
        "vector": vector,
        "topK": 1,
        "includeMetadata": true
    });

    let res = client
        .post(index_url)
        .header("Api-Key", api_key)
        .json(&body)
        .send()
        .await?
        .json::<QueryResponse>()
        .await?;

    Ok(res.matches)
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    let pinecone_key = env::var("PINECONE_API_KEY")?;
    let pinecone_url = env::var("PINECONE_INDEX_URL")?; // e.g. https://your-index.svc.pinecone.io/query

    let client = Client::new();

    // Load tokenizer + ONNX model
    let tokenizer = Tokenizer::from_file("onnx_model/tokenizer.json").map_err(|e| anyhow!(e))?;
    let environment = Arc::new(Environment::builder().with_name("embedder").build()?);
    let session = SessionBuilder::new(&environment)?
        .with_model_from_file("onnx_model/model.onnx")?;

    // Example queries
    let queries = vec![
        "I need a new nextjs project",
        "please give me a nextjs boilerplate",
    ];

    for query in queries {
        println!("\n🔎 Query: {}", query);

        let embedding = embed_text(&session, &tokenizer, query)?;
        let results = query_pinecone(&client, &pinecone_key, &pinecone_url, embedding).await?;

        for m in results {
            println!(
                "➡ Command: {:?}, Score: {}",
                m.metadata
                    .as_ref()
                    .and_then(|md| md.get("command"))
                    .unwrap_or(&serde_json::Value::String("N/A".to_string())),
                m.score
            );
        }
    }

    Ok(())
}