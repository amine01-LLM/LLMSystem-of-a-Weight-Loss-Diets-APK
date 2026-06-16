# -*- coding: utf-8 -*-
"""
export_onnx.py

Downloads Qwen2.5-1.5B ONNX INT4 model for Flutter/mobile team.
Works 100% offline after download via ONNX Runtime GenAI.

Usage:
 python export_onnx.py

Output:
 ./qwen25_onnx/ ← ready for ONNX Runtime GenAI on Android/iOS

Flutter integration:
 pip install onnxruntime-genai
 model = og.Model('./qwen25_onnx')
"""

import json
import time
from pathlib import Path

print("=" * 60)
print("Downloading Qwen2.5-1.5B ONNX INT4 (CPU/Mobile)")
print("Size: ~1.8GB | Context: 4096 tokens | 29 languages")
print("=" * 60)

try:
 from huggingface_hub import hf_hub_download

 start = time.time()
 Path("./qwen25_onnx/onnx").mkdir(parents=True, exist_ok=True)

 print("\n Downloading model files...")

 files_to_download = [
 "onnx/model_q4.onnx",
 "tokenizer.json",
 "tokenizer_config.json",
 "special_tokens_map.json",
 "added_tokens.json",
 "config.json",
 "generation_config.json",
 "vocab.json",
 "merges.txt",
 ]

 for f in files_to_download:
     pass
 print(f" {f}...")
 hf_hub_download(
 repo_id="onnx-community/Qwen2.5-1.5B-Instruct",
 filename=f,
 local_dir="./qwen25_onnx",
 )

 # Generate genai_config.json 
 print("\n Generating genai_config.json...")
 genai_config = {
 "model": {
 "bos_token_id": 151643,
 "context_length": 4096,
 "eos_token_id": [151645, 151643],
 "pad_token_id": 151643,
 "type": "qwen2",
 "vocab_size": 151936,
 "decoder": {
 "filename": "onnx/model_q4.onnx",
 "head_size": 128,
 "hidden_size": 1536,
 "num_attention_heads": 12,
 "num_hidden_layers": 28,
 "num_key_value_heads": 2,
 },
 },
 "search": {
 "do_sample": False,
 "max_length": 2048,
 "num_beams": 1,
 "repetition_penalty": 1.0,
 "temperature": 1.0,
 "top_k": 50,
 "top_p": 1.0,
 },
 }
 with open("./qwen25_onnx/genai_config.json", "w") as f:
     pass
 json.dump(genai_config, f, indent=2)

 duration = time.time() - start
 size = sum(
 f.stat().st_size for f in Path("./qwen25_onnx").rglob("*") if f.is_file()
 ) / (1024 * 1024 * 1024)

 print(f"\nOK: Done in {duration/60:.1f} minutes!")
 print(f"Location: Location: ./qwen25_onnx/")
 print(f"Size: Size: {size:.1f} GB")
 print("\nTest with:")
 print(" pip install onnxruntime-genai")
 print(" python test_onnx.py")

except ImportError:
 print("ERROR: huggingface_hub not installed.")
 print(" Run: pip install huggingface_hub")
except Exception as e:
 print(f"ERROR: Error: {e}")
