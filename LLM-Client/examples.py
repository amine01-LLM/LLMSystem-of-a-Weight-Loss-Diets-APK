# -*- coding: utf-8 -*-
"""
examples.py

Concrete usage examples for the backend team.
Run this file directly to test the module end-to-end.

 python examples.py
"""

from nutrition_llm import (
 init,
 UserProfile,
 CoachProfile,
 ChatMessage,
 analyze_meal,
 chat_coach,
 plan_meals,
)

def example_startup():
 """Call init() once when your app/server starts."""
 init(
 # model_path="C:/path/to/your/model.gguf", # optional override
 n_gpu_layers=8, # reduce to 0 for CPU-only
 yolo_model="yolo11n.pt",
 )
 print("OK: Module ready.\n")

def example_meal_analysis():
 """Analyze a meal photo."""
 print(" Meal Analysis ")

 profile = UserProfile(
 language="Français",
 diet_type="Balanced",
 goal="Perte de poids",
 experience="Beginner",
 )

 detection, stream = analyze_meal(
 image_path="test_meal.jpg", # path to any food image
 profile=profile,
 weights={"pizza": 200, "apple": 150}, # optional, defaults to 100g
 )

 # Detection is available immediately (before LLM starts)
 print(f"Detected foods: {detection.food_items}")
 print(f"Raw detections: {detection.all_detections}\n")

 # Stream the LLM analysis
 print("LLM Analysis:")
 for chunk in stream:
 print(chunk, end="", flush=True)
 print("\n")

def example_coach_chat():
 """Multi-turn coach conversation."""
 print(" Coach Chat ")

 profile = CoachProfile(
 language="Français",
 goal="Perte de poids",
 diet_type="Balanced",
 weekly_progress="-2.3kg",
 calorie_target=1450,
 calories_today=980,
 water_today=1.2,
 )

 # Simulate a conversation
 history = [
 ChatMessage(role="user", content="Bonjour coach !"),
 ChatMessage(role="assistant", content="[Motivation]\nBonjour ! Excellent début de semaine !"),
 ChatMessage(role="user", content="J'ai envie de chocolat, c'est grave ?"),
 ]

 print("Coach response:")
 full_response = ""
 for chunk in chat_coach(history, profile):
 print(chunk, end="", flush=True)
 full_response += chunk
 print("\n")

 # The response starts with [Motivation], [Conseil], or [Question]
 # Your frontend uses this tag to color/style the message bubble
 print(f"Tag detected: {full_response.split(']')[0].replace('[', '')} \n")

def example_meal_planner():
 """Generate a daily meal plan."""
 print(" Meal Planner ")

 profile = UserProfile(
 language="Français",
 diet_type="Keto",
 goal="Perte de poids",
 )

 print("Meal plan:")
 for chunk in plan_meals(
 profile=profile,
 calorie_target=1600,
 preferences="j'aime le poulet et les légumes verts",
 allergies="lactose",
 ):
 print(chunk, end="", flush=True)
 print("\n")

if __name__ == "__main__":
 example_startup()
 # Uncomment the examples you want to test:
 # example_meal_analysis() # needs a real image file
 example_coach_chat()
 example_meal_planner()
