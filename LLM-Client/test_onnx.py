# -*- coding: utf-8 -*-
"""
test_onnx.py

Quick test for Qwen2.5 ONNX model.
Run after export_onnx.py to verify the model works.

Usage:
 python test_onnx.py
"""

import time

try:
 import onnxruntime_genai as og
except ImportError:
 print("ERROR: onnxruntime-genai not installed.")
 print(" Run: pip install onnxruntime-genai")
 exit(1)

MODEL_PATH = "./qwen25_onnx"

def generate(system: str, user: str, max_length: int = 300) -> str:
 prompt = (
 f"<|im_start|>system\n{system}<|im_end|>\n"
 f"<|im_start|>user\n{user}<|im_end|>\n"
 f"<|im_start|>assistant\n"
 )
 params = og.GeneratorParams(model)
 params.set_search_options(max_length=max_length, temperature=0.7)
 generator = og.Generator(model, params)
 generator.append_tokens(tokenizer.encode(prompt))

 response = ""
 while not generator.is_done():
     pass
 generator.generate_next_token()
 token = generator.get_next_tokens()[0]
 response += tokenizer.decode([token])
 return response.strip()

print("=" * 60)
print("Qwen2.5 ONNX — Test Suite")
print("=" * 60)

print("\n Loading model...")
start = time.time()
model = og.Model(MODEL_PATH)
tokenizer = og.Tokenizer(model)
print(f"OK: Loaded in {time.time()-start:.1f}s")

system = "You are a nutrition coach. Be concise and motivating."

# Test 1 — English
print("\nTest Test 1 — English:")
start = time.time()
r = generate(system, "how to lose weight fast?", max_length=150)
print(f"{r[:200]}...")
print(f"Time: {time.time()-start:.1f}s")

# Test 2 — French
print("\nTest Test 2 — French:")
start = time.time()
r = generate(system, "comment perdre du poids rapidement?", max_length=150)
print(f"{r[:200]}...")
print(f"Time: {time.time()-start:.1f}s")

# Test 3 — Arabic
print("\nTest Test 3 — Arabic:")
start = time.time()
r = generate(system, "كيف أخسر الوزن بسرعة؟", max_length=150)
print(f"{r[:200]}...")
print(f"Time: {time.time()-start:.1f}s")

print("\nOK: All tests passed!")
