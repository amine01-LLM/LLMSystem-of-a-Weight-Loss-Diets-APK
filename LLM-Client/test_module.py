# -*- coding: utf-8 -*-
"""
test_module.py

Run this to verify the LLM module works end-to-end.
Place this file inside the llm_module/ folder and run:

 python test_module.py
"""

import sys
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

print("\n" + "="*50)
print(" AGH Nutrition AI — Module Test")
print("="*50 + "\n")

# Step 1: Import 
print(" Step 1: Importing module...")
try:
 from nutrition_llm import (
 init, UserProfile, CoachProfile,
 ChatMessage, analyze_meal, chat_coach, plan_meals
 )
 print("OK: Import OK\n")
except Exception as e:
 print(f"ERROR: Import failed: {e}")
 sys.exit(1)

# Step 2: Init 
print(" Step 2: Initializing engines...")
try:
 init(n_gpu_layers=8, yolo_model="yolo11n.pt")
 print("OK: Engines loaded\n")
except Exception as e:
 print(f"ERROR: Init failed: {e}")
 sys.exit(1)

# Step 3: Coach Chat 
print(" Step 3: Testing Coach Chat...")
print("-" * 40)
try:
 profile = CoachProfile(
 language="Français",
 goal="Perte de poids",
 diet_type="Balanced",
 weekly_progress="-2.3kg",
 calorie_target=1450,
 calories_today=980,
 water_today=1.2,
 )

 history = [
 ChatMessage(role="user", content="Bonjour coach, j'ai bien mangé aujourd'hui !"),
 ]

 full_response = ""
 for chunk in chat_coach(history, profile):
 print(chunk, end="", flush=True)
 full_response += chunk

 print("\n" + "-" * 40)

 # Check tag
 import re
 match = re.match(r'^\[(Motivation|Conseil|Question)\]', full_response.strip())
 if match:
 print(f"OK: Tag detected: [{match.group(1)}]\n")
 else:
 print("WARNING: No tag found in response (model may need tuning)\n")

except Exception as e:
 print(f"\nERROR: Coach chat failed: {e}\n")

# Step 4: Meal Planner 
print(" Step 4: Testing Meal Planner...")
print("-" * 40)
try:
 profile = UserProfile(
 language="Français",
 diet_type="Balanced",
 goal="Perte de poids",
 )

 full_plan = ""
 for chunk in plan_meals(
 profile=profile,
 calorie_target=1600,
 preferences="poulet et légumes",
 allergies="None",
 ):
 print(chunk, end="", flush=True)
 full_plan += chunk

 print("\n" + "-" * 40)
 print("OK: Meal planner OK\n")

except Exception as e:
 print(f"\nERROR: Meal planner failed: {e}\n")

# Step 5: Vision (optional) 
print(" Step 5: Testing Vision Detection (skipped — no image provided)")
print(" To test vision, run:")
print(" >>> from nutrition_llm import analyze_meal, UserProfile")
print(" >>> d, s = analyze_meal('your_image.jpg', UserProfile())")
print(" >>> print(d.food_items)")
print()

print("="*50)
print(" All tests complete!")
print("="*50 + "\n")
