# Data Team API Contract
## LLM_Client ↔ Data Team

This document defines the API that the **Data team** must provide
so the **LLM_Client** module can inject live user stats into the coach.

---

## Endpoint Required

```
GET /api/user-stats/{user_id}
```

### Request
| Parameter | Type | Description |
|-----------|--------|----------------------|
| `user_id` | string | Unique user ID from auth system |

### Response (200 OK)
```json
{
 "user_id": "user_42",
 "date": "2026-02-23",
 "calories_today": 1250,
 "water_today": 1.5,
 "weekly_progress": "-2.3kg"
}
```

### Response fields
| Field | Type | Unit | Description |
|-------------------|--------|---------|------------------------------------|
| `user_id` | string | — | Echo of the requested user ID |
| `date` | string | ISO date| Date of the stats |
| `calories_today` | int | kcal | Total calories consumed today |
| `water_today` | float | litres | Total water consumed today |
| `weekly_progress` | string | kg | Weight change this week e.g "-2.3kg" |

### Error responses
```json
{ "error": "user not found" } // 404
{ "error": "server error" } // 500
```

---

## Behavior Contract

- If user has no data yet → return `calories_today: 0, water_today: 0.0, weekly_progress: "0kg"`
- Response time must be **< 2 seconds** (LLM_Client has a 2s timeout)
- If API is unreachable → LLM_Client falls back to defaults silently
- Data resets daily at **midnight local time**

---

## Example (Python test)
```python
import requests

resp = requests.get("http://data-service/api/user-stats/user_42")
print(resp.json())
# {
# "user_id": "user_42",
# "date": "2026-02-23",
# "calories_today": 1250,
# "water_today": 1.5,
# "weekly_progress": "-2.3kg"
# }
```

---

## How LLM_Client uses this data

The values are injected into the coach system prompt:

```
User stats today:
- Calories consumed: 1250 / 1600 kcal target
- Water: 1.5L
- Weekly progress: -2.3kg
```

This allows the coach to give personalized advice like:
> "You still have 350 kcal left today — a grilled chicken salad would be perfect!"
