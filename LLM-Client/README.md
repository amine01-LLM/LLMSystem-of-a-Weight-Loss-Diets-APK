# AGH Nutrition AI — LLM_Client Module

> **Cahier de charge:** AGH Data Agency Holding — APK Régime & Perte de Poids
> **Responsible:** Bilal
> **Architecture:** 95% client-side / 5% server-side — 100% offline after setup

---

## File Structure

```
LLM_Client/
├── nutrition_llm.py          ← PUBLIC INTERFACE (import from here)
├── llm_engine.py             ← Qwen2.5 ONNX singleton (internal)
├── vision_engine.py          ← YOLO ONNX singleton (internal)
├── memory.py                 ← SQLite persistent memory (internal)
├── django_integration.py     ← Django views (copy into Django app)
├── export_onnx.py            ← Download Qwen2.5 ONNX model
├── test_onnx.py              ← Test LLM in EN/FR/AR
├── data_team_api.md          ← API contract for data team
├── HANDOFF.md                ← Setup guide for all teams
├── prompts/
│   ├── coach_system.md
│   ├── meal_analysis.md
│   └── meal_planner.md
├── flutter_reference/
│   └── memory.dart           ← Flutter/Dart equivalent of memory.py
├── models/
│   └── best_food256.onnx     ← YOLO food detection (256 classes)
├── examples.py
├── requirements.txt
└── README.md
```

---

## Models Required

| Model | Format | Size | Purpose |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | ONNX INT4 | ~1.8GB | LLM coach + planner |
| best_food256 | ONNX | ~62MB | Food detection (256 classes) |

```
project_root/
├── LLM_Client/          ← this folder
├── qwen25_onnx/         ← Qwen2.5 ONNX model (downloaded via export_onnx.py)
└── models/
    └── best_food256.onnx
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download Qwen2.5 ONNX model (~1.8GB, one time only)**
```bash
python export_onnx.py
```

**3. Test everything works**
```bash
python test_onnx.py      # test LLM in EN/FR/AR
python interactive.py    # full CLI test
```

---

## Quick Start (Django backend)

```python
from nutrition_llm import init, UserProfile, CoachProfile, ChatMessage
from nutrition_llm import analyze_meal, chat_coach, plan_meals

# Call ONCE when Django server starts
init(
    model_path="qwen25_onnx",
    yolo_model="models/best_food256.onnx"
)
```

---

## API Reference

### `init(model_path, yolo_model)`

Loads both LLM and Vision models. Must be called once at startup.

```python
init()                                          # all defaults
init(model_path="qwen25_onnx")                 # custom LLM path
init(yolo_model="models/best_food256.onnx")    # custom YOLO path
```

---

### `chat_coach(messages, profile, user_id)` → `Generator[str]`

Coach chat with persistent memory across sessions.

```python
profile = CoachProfile(
    language="Français",
    goal="Perte de poids",
    calorie_target=1450,
    calories_today=800,
    water_today=1.0,
    weekly_progress="-2.3kg",
)

for chunk in chat_coach(
    [ChatMessage(role="user", content="J'ai mange une pizza")],
    profile,
    user_id="user_42"
):
    send_to_client(chunk)
```

**Response tags (first chunk always starts with one):**

| Tag | Meaning | UI color |
|---|---|---|
| `[Motivation]` | Encouragement | Green |
| `[Conseil]` | Actionable advice | Orange |
| `[Question]` | Coach asks user | Red |

---

### `analyze_meal(image_path, profile, weights)` → `(MealDetectionResult, Generator)`

```python
profile = UserProfile(language="Français", diet_type="Keto", goal="Perte de poids")

detection, stream = analyze_meal("meal.jpg", profile)

print(detection.food_items)    # ['steak', 'green salad'] — immediate
for chunk in stream:           # stream analysis to client
    send_to_client(chunk)
```

Note: Image is analyzed then immediately discarded — never stored (RGPD compliant).

---

### `plan_meals(profile, calorie_target, preferences, allergies)` → `Generator[str]`

```python
for chunk in plan_meals(
    profile,
    calorie_target=1600,
    preferences="poulet, legumes verts",
    allergies="lactose"
):
    send_to_client(chunk)
```

---

## Data Classes

```python
@dataclass
class UserProfile:
    language:   str = "Francais"    # see supported languages below
    diet_type:  str = "Balanced"    # "Keto" | "Vegan" | "Mediterranean"
    goal:       str = "Healthy eating"
    experience: str = "Beginner"

@dataclass
class CoachProfile(UserProfile):
    weekly_progress: str   = "0kg"
    calorie_target:  int   = 2000
    calories_today:  int   = 0
    water_today:     float = 0.0    # litres

@dataclass
class ChatMessage:
    role:    str    # "user" | "assistant"
    content: str
```

---

## Supported Languages

| Value | Language |
|---|---|
| `"Francais"` | French |
| `"English"` | English |
| `"Arabic"` | Modern Standard Arabic |
| `"Espanol"` | Spanish |
| `"Italiano"` | Italian |
| `"Portugues"` | Portuguese |
| `"Deutsch"` | German |
| `"Chinese"` | Chinese |

---

## Memory System

Conversation memory is persistent across sessions via SQLite.

```python
# Backend passes user_id from auth system
chat_coach(messages, profile, user_id="user_42")

# Module automatically:
# - Saves every message to conversations.db
# - Summarizes every 20 messages
# - Injects summary + history into every prompt
```

**Data team provides per request:**
```json
{
    "user_id":         "user_42",
    "calories_today":  1250,
    "water_today":     1.5,
    "weekly_progress": "-2.3kg"
}
```

---

## Django Integration

See `django_integration.py` — copy into your Django app and wire up urls.py:

```python
from django_integration import coach_view, meal_view, planner_view, stats_view

urlpatterns = [
    path('api/coach/',   coach_view),
    path('api/meal/',    meal_view),
    path('api/planner/', planner_view),
    path('api/stats/',   stats_view),
]
```

---

## Flutter Team Reference

See `flutter_reference/memory.dart` for the Dart equivalent of `memory.py`.

**Flutter dependencies:**
```yaml
dependencies:
    sqflite: ^2.3.0
    path: ^1.9.0
    onnxruntime: ^1.0.0
```

**Mobile models:**
- LLM: `qwen25_onnx/` via `onnxruntime-genai`
- Vision: `best_food256.onnx` via `onnxruntime`

---

## Performance

| Feature | PC (CPU) | Phone GPU (estimated) |
|---|---|---|
| Model load | ~6s | ~8s |
| Coach response | ~9s | ~3-4s |
| Meal scan | ~1s | less than 1s |
| Meal plan | ~25s | ~8-10s |

---

## Error Reference

| Exception | Cause | Fix |
|---|---|---|
| `RuntimeError: LLM engine not initialized` | `init()` not called | Call `init()` at startup |
| `FileNotFoundError: Qwen2.5 ONNX not found` | Model not downloaded | Run `python export_onnx.py` |
| `FileNotFoundError: Image not found` | Bad image path | Verify path exists |
| `ValueError: Last message must be role=user` | Wrong message order | Ensure last message is user |
