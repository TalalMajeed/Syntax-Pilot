from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

# 1. Choose the model (you can replace with any sentence-transformers model)
model_id = "sentence-transformers/all-MiniLM-L6-v2"

# 2. Path to save ONNX model
onnx_path = "./onnx_model"

# 3. Load and export model
print(f"Exporting {model_id} to ONNX...")
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
model.save_pretrained(onnx_path)

# 4. Save tokenizer (needed in Rust for preprocessing text)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(onnx_path)

print(f"✅ Model exported successfully to: {onnx_path}")