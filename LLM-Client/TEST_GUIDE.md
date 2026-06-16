# LLM_Client — Test Guide
## AGH Data Agency Holding — APK Regime & Perte de Poids

---

## Prerequisites

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Download Qwen2.5 model (~1.8GB, one time only):**
```bash
python export_onnx.py
```

**Verify models are in place:**
```bash
# Linux / Mac
ls qwen25_onnx/
ls models/best_food256.onnx

# Windows
dir qwen25_onnx\
dir models\best_food256.onnx
```

Models required:
- `qwen25_onnx/` — downloaded via export_onnx.py
- `models/best_food256.onnx` — already in repository (62MB)

---

## Test 1 — LLM works in 3 languages

```bash
python test_onnx.py
```

Expected output:
```
Loaded in ~6s
Test 1 - English: response about nutrition in English
Test 2 - French: response in French
Test 3 - Arabic: response in Arabic
All tests passed!
```

Pass criteria:
- All 3 responses generated without error
- Each response is coherent and on-topic
- Response time under 30s per language (CPU)

---

## Test 2 — Full CLI interface

```bash
python interactive.py
```

Steps to test:
1. Select language (test at least: 1=French, 2=English, 3=Arabic)
2. Select option 1 (Coach chat) — send a message, verify response has tag [Motivation], [Conseil], or [Question]
3. Select option 3 (Meal planner) — verify a full meal plan is generated
4. Type exit/quit — verify app exits cleanly

Pass criteria:
- Language picker works correctly
- Coach response always starts with [Motivation], [Conseil], or [Question]
- Meal plan contains Breakfast, Lunch, Dinner sections
- No crash or error during normal use

---

## Test 3 — Coach API

```python
from nutrition_llm import init, CoachProfile, ChatMessage, chat_coach

init(model_path="qwen25_onnx", yolo_model="models/best_food256.onnx")

profile = CoachProfile(
    language="Francais",
    goal="Perte de poids",
    calorie_target=1600,
    calories_today=800,
    water_today=1.0,
    weekly_progress="-1.5kg",
)

response = "".join(chat_coach(
    [ChatMessage(role="user", content="J'ai mange une pizza ce midi")],
    profile,
    user_id="test_user_1"
))

assert response.startswith(("[Motivation]", "[Conseil]", "[Question]")), "FAIL: Missing tag"
assert len(response) > 20, "FAIL: Response too short"
print("PASS:", response[:150])
```

Pass criteria:
- No exception raised
- Response starts with [Motivation], [Conseil], or [Question]
- Response is in French

---

## Test 4 — Coach streaming

Verify that the response streams token by token, not all at once.

```python
from nutrition_llm import init, CoachProfile, ChatMessage, chat_coach
import time

init(model_path="qwen25_onnx", yolo_model="models/best_food256.onnx")

profile = CoachProfile(language="English", goal="Healthy eating")

chunks = []
for chunk in chat_coach(
    [ChatMessage(role="user", content="Give me a tip")],
    profile,
    user_id="test_stream"
):
    chunks.append(chunk)
    print(chunk, end="", flush=True)

print()
assert len(chunks) > 1, "FAIL: Response arrived in one block, not streaming"
print(f"PASS: streamed in {len(chunks)} chunks")
```

Pass criteria:
- `len(chunks) > 1` — text arrives progressively
- No error during streaming

---

## Test 5 — Food detection (Vision)

```python
from nutrition_llm import init
from vision_engine import detect_food

init(model_path="qwen25_onnx", yolo_model="models/best_food256.onnx")

result = detect_food("path/to/any/food/image.jpg")

assert "food_items" in result, "FAIL: food_items key missing"
assert "all_detections" in result, "FAIL: all_detections key missing"
assert isinstance(result["food_items"], list), "FAIL: food_items is not a list"
print("PASS:", result["food_items"])
```

Pass criteria:
- No exception raised
- Returns dict with food_items and all_detections keys
- food_items is a list (can be empty if no food detected)

---

## Test 6 — Meal analysis API

```python
from nutrition_llm import init, UserProfile, analyze_meal

init(model_path="qwen25_onnx", yolo_model="models/best_food256.onnx")

profile = UserProfile(language="English", diet_type="Balanced", goal="Healthy eating")

detection, stream = analyze_meal("path/to/food/image.jpg", profile)

assert hasattr(detection, "food_items"), "FAIL: food_items missing"
assert hasattr(detection, "all_detections"), "FAIL: all_detections missing"

analysis = "".join(stream)
assert len(analysis) > 20, "FAIL: analysis too short"
print("PASS food_items:", detection.food_items)
print("PASS analysis:", analysis[:100])
```

Pass criteria:
- detection.food_items available immediately before stream starts
- analysis streams without error
- analysis text is coherent

---

## Test 7 — Memory persistence

```python
from memory import init_db, save_message, load_recent_messages, clear_user_history

init_db()

# Clean slate
clear_user_history("test_memory_user")

# Save messages
save_message("test_memory_user", "user", "Je veux perdre du poids")
save_message("test_memory_user", "assistant", "Je peux vous aider avec ca!")

# Load back
messages = load_recent_messages("test_memory_user")

assert len(messages) == 2, "FAIL: expected 2 messages"
assert messages[0]["role"] == "user", "FAIL: wrong role order"
assert messages[1]["role"] == "assistant", "FAIL: wrong role order"
print("PASS: memory saved and loaded correctly")
```

Pass criteria:
- Messages saved and retrieved correctly
- Order preserved (oldest first)
- No database error

---

## Test 8 — Multi-language responses

```python
from nutrition_llm import init, CoachProfile, ChatMessage, chat_coach

init(model_path="qwen25_onnx", yolo_model="models/best_food256.onnx")

languages = ["Francais", "English", "Arabic", "Espanol", "Italiano", "Portugues", "Deutsch", "Chinese"]
message = "Give me a nutrition tip"

for lang in languages:
    profile = CoachProfile(language=lang, goal="Healthy eating")
    response = "".join(chat_coach(
        [ChatMessage(role="user", content=message)],
        profile,
        user_id=f"test_{lang}"
    ))
    assert len(response) > 10, f"FAIL: empty response for {lang}"
    print(f"PASS {lang}: {response[:80]}")
```

Pass criteria:
- All 8 languages generate a response
- Arabic responds in Arabic script
- No language falls back to English unexpectedly

---

## Test 9 — Offline mode

1. Disconnect internet completely (airplane mode or disable network)
2. Run:
```bash
python interactive.py
```

Pass criteria:
- App loads without error
- Coach chat works
- Meal planner works
- No internet connection error

---

## Known Limitations

| Item | Detail |
|---|---|
| Response time | 8-10s on CPU — normal, will be 3-4x faster on phone GPU |
| Food detection accuracy | Current model trained mainly on Asian food — accuracy on Western food improves after merged training completes |
| Model load time | ~6s on first call — normal |
| YOLO confidence threshold | Set to 0.20 — may detect incorrect food on low quality images |

---

## Reporting Issues

For each failed test, include:
1. Test number and name
2. Full error message and traceback
3. Python version (`python --version`)
4. OS (Windows / Mac / Linux)
5. Output of model check:
   - Linux/Mac: `ls qwen25_onnx/`
   - Windows: `dir qwen25_onnx\`
