# LLM_Client — Final Handoff
## AGH Data Agency Holding — APK Régime & Perte de Poids

---

## For the Django Backend Team

### 1. Setup

```bash
# Install dependencies
pip install -r LLM_Client/requirements.txt

# Download Qwen2.5 ONNX model (~1.8GB, one time only)
cd LLM_Client
python export_onnx.py
```

### 2. Django settings.py

```python
import sys
sys.path.append(BASE_DIR / 'LLM_Client')
```

### 3. Django apps.py

```python
from django.apps import AppConfig

class YourAppConfig(AppConfig):
 name = 'your_app'

 def ready(self):
 from nutrition_llm import init
 init(
 model_path='qwen25_onnx',
 yolo_model='LLM_Client/models/best_food256.onnx'
 )
```

### 4. urls.py

```python
from django_integration import coach_view, meal_view, planner_view, stats_view

urlpatterns = [
 path('api/coach/', coach_view),
 path('api/meal/', meal_view),
 path('api/planner/', planner_view),
 path('api/stats/', stats_view),
]
```

### 5. Update data team API URL

In `django_integration.py`, line 28:
```python
DATA_TEAM_API = "http://YOUR_DATA_SERVICE_URL/api/user-stats"
```

---

## For the Flutter Team

### Models needed (download once)
```bash
python LLM_Client/export_onnx.py # downloads qwen25_onnx/ (~1.8GB)
```
YOLO model: `LLM_Client/models/best_food256.onnx` (62MB — in git)

### Memory system
See `LLM_Client/flutter_reference/memory.dart`
Same logic as Python `memory.py` — just uses `sqflite` instead of `sqlite3`.

### pubspec.yaml
```yaml
dependencies:
 sqflite: ^2.3.0
 path: ^1.9.0
 onnxruntime: ^1.0.0
```

### LLM inference
```dart
import 'package:onnxruntime_genai/onnxruntime_genai.dart';

final model = OrtGenAIModel('assets/qwen25_onnx');
final tokenizer = OrtGenAITokenizer(model);
```

---

## For the Data Team

See `LLM_Client/data_team_api.md` for full API contract.

**Summary:** Implement this endpoint:
```
GET /api/user-stats/{user_id}

Response:
{
 "calories_today": 1250,
 "water_today": 1.5,
 "weekly_progress": "-2.3kg"
}
```

---

## File Structure

```
LLM_Client/
 nutrition_llm.py ← PUBLIC API (Django team imports from here)
 django_integration.py ← Django views (copy into Django app)
 llm_engine.py ← Qwen2.5 ONNX engine (internal)
 vision_engine.py ← YOLO ONNX engine (internal)
 memory.py ← SQLite memory (internal)
 export_onnx.py ← Download Qwen2.5 model
 test_onnx.py ← Test LLM in EN/FR/AR
 data_team_api.md ← API contract for data team
 HANDOFF.md ← This file
 prompts/
 coach_system.md
 meal_analysis.md
 meal_planner.md
 flutter_reference/
 memory.dart ← Flutter memory implementation
 models/
 best_food256.onnx ← YOLO food detection model
 requirements.txt
 README.md
```

---

## API Endpoints Summary

| Endpoint | Method | Description |
|---|---|---|
| `/api/coach/` | POST | Coach chat with memory |
| `/api/meal/` | POST | Food detection + analysis |
| `/api/planner/` | POST | Daily meal plan |
| `/api/stats/` | GET | User stats proxy |

---

## Performance (PC CPU — phone will be faster)

| Feature | Time |
|---|---|
| Model load (once) | ~6s |
| Coach response | ~9s |
| Meal scan | ~1s |
| Meal plan | ~25s |

---

## Pending (after Kaggle training finishes)

- [ ] Download `best_food_merged_final.pt` from Kaggle output
- [ ] Export to `best_food_merged_final.onnx`
- [ ] Replace `models/best_food256.onnx` with merged model (336 classes)
- [ ] Update `FOOD_CLASSES` dict in `vision_engine.py` with 336 class names
