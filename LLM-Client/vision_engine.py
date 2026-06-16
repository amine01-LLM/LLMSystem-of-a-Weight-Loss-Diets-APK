# -*- coding: utf-8 -*-
"""
vision_engine.py

YOLO food detection via ONNX Runtime.
Supports:
- ONNX models (recommended for production)
- Ultralytics .pt fallback (for development)

Designed for singleton model loading (load once, reuse everywhere).
"""

import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# UEC Food-256 Classes
# ---------------------------------------------------------------------

FOOD_CLASSES = {
 0: "rice", 1: "eels on rice", 2: "pilaf", 3: "chicken-'n'-egg on rice",
 4: "pork cutlet on rice", 5: "beef curry", 6: "sushi", 7: "chicken rice",
 8: "fried rice", 9: "tempura bowl", 10: "bibimbap", 11: "toast",
 12: "croissant", 13: "roll bread", 14: "raisin bread", 15: "chip butty",
 16: "hamburger", 17: "pizza", 18: "sandwiches", 19: "udon noodle",
 20: "tempura udon", 21: "soba noodle", 22: "ramen noodle", 23: "beef noodle",
 24: "tensin noodle", 25: "fried noodle", 26: "spaghetti", 27: "Japanese-style pancake",
 28: "takoyaki", 29: "gratin", 30: "sauteed vegetables", 31: "croquette",
 32: "grilled eggplant", 33: "sauteed spinach", 34: "vegetable tempura",
 35: "miso soup", 36: "potage", 37: "sausage", 38: "oden", 39: "omelet",
 40: "ganmodoki", 41: "jiaozi", 42: "stew", 43: "teriyaki grilled fish",
 44: "fried fish", 45: "grilled salmon", 46: "salmon meuniere", 47: "sashimi",
 48: "grilled pacific saury", 49: "sukiyaki", 50: "sweet and sour pork",
 51: "lightly roasted fish", 52: "steamed egg hotchpotch", 53: "tempura",
 54: "fried chicken", 55: "sirloin cutlet", 56: "nanbanzuke", 57: "boiled fish",
 58: "seasoned beef with potatoes", 59: "hambarg steak", 60: "steak",
 61: "dried fish", 62: "ginger pork saute", 63: "spicy chili-flavored tofu",
 64: "yakitori", 65: "cabbage roll", 66: "omelet", 67: "egg sunny-side up",
 68: "natto", 69: "cold tofu", 70: "egg roll", 71: "chilled noodle",
 72: "stir-fried beef and peppers", 73: "simmered pork",
 74: "boiled chicken and vegetables", 75: "sashimi bowl", 76: "sushi bowl",
 77: "fish-shaped pancake with bean jam", 78: "shrimp with chill source",
 79: "roast chicken", 80: "steamed meat dumpling", 81: "omelet with fried rice",
 82: "cutlet curry", 83: "spaghetti meat sauce", 84: "fried shrimp",
 85: "potato salad", 86: "green salad", 87: "macaroni salad",
 88: "Japanese tofu and vegetable chowder", 89: "pork miso soup",
 90: "chinese soup", 91: "beef bowl", 92: "kinpira-style sauteed burdock",
 93: "rice ball", 94: "pizza toast", 95: "dipping noodles", 96: "hot dog",
 97: "french fries", 98: "mixed rice", 99: "goya chanpuru", 100: "green curry",
 101: "okinawa soba", 102: "mango pudding", 103: "almond jelly", 104: "jjigae",
 105: "dak galbi", 106: "dry curry", 107: "kamameshi", 108: "rice vermicelli",
 109: "paella", 110: "tanmen", 111: "kushikatu", 112: "yellow curry",
 113: "pancake", 114: "champon", 115: "crape", 116: "tiramisu", 117: "waffle",
 118: "rare cheese cake", 119: "shortcake", 120: "chop suey",
 121: "twice cooked pork", 122: "mushroom risotto", 123: "samul", 124: "zoni",
 125: "french toast", 126: "fine white noodles", 127: "minestrone",
 128: "pot au feu", 129: "chicken nugget", 130: "namero", 131: "french bread",
 132: "rice gruel", 133: "broiled eel bowl", 134: "clear soup", 135: "yudofu",
 136: "mozuku", 137: "inarizushi", 138: "pork loin cutlet",
 139: "pork fillet cutlet", 140: "chicken cutlet", 141: "ham cutlet",
 142: "minced meat cutlet", 143: "thinly sliced raw horsemeat", 144: "bagel",
 145: "scone", 146: "tortilla", 147: "tacos", 148: "nachos", 149: "meat loaf",
 150: "scrambled egg", 151: "rice gratin", 152: "lasagna", 153: "Caesar salad",
 154: "oatmeal", 155: "fried pork dumplings served in soup", 156: "oshiruko",
 157: "muffin", 158: "popcorn", 159: "cream puff", 160: "doughnut",
 161: "apple pie", 162: "parfait", 163: "fried pork in scoop",
 164: "lamb kebabs", 165: "stir-fried potato eggplant and green pepper",
 166: "roast duck", 167: "hot pot", 168: "pork belly", 169: "xiao long bao",
 170: "moon cake", 171: "custard tart", 172: "beef noodle soup",
 173: "pork cutlet", 174: "minced pork rice", 175: "fish ball soup",
 176: "oyster omelette", 177: "glutinous oil rice", 178: "turnip pudding",
 179: "stinky tofu", 180: "lemon fig jelly", 181: "khao soi",
 182: "sour prawn soup", 183: "Thai papaya salad",
 184: "Hainan chicken with marinated rice", 185: "hot and sour fish ragout",
 186: "stir-fried mixed vegetables", 187: "beef in oyster sauce",
 188: "pork satay", 189: "spicy chicken salad", 190: "noodles with fish curry",
 191: "pork sticky noodles", 192: "pork with lemon", 193: "stewed pork leg",
 194: "charcoal-boiled pork neck", 195: "fried mussel pancakes",
 196: "deep fried chicken wing", 197: "barbecued red pork with rice",
 198: "rice with roast duck", 199: "rice crispy pork", 200: "wonton soup",
 201: "chicken rice curry with coconut", 202: "crispy noodles",
 203: "egg noodle in chicken yellow curry", 204: "coconut milk soup",
 205: "pho", 206: "Hue beef rice vermicelli soup",
 207: "vermicelli noodles with snails", 208: "fried spring rolls",
 209: "steamed rice roll", 210: "shrimp patties", 211: "bun with pork",
 212: "coconut crepes with shrimp and beef", 213: "steamed savory rice pancake",
 214: "glutinous rice balls", 215: "loco moco", 216: "haupia",
 217: "malasada", 218: "laulau", 219: "spam musubi", 220: "oxtail soup",
 221: "adobo", 222: "lumpia", 223: "brownie", 224: "churro",
 225: "jambalaya", 226: "nasi goreng", 227: "ayam goreng", 228: "ayam bakar",
 229: "bubur ayam", 230: "gulai", 231: "laksa", 232: "mie ayam",
 233: "mie goreng", 234: "nasi campur", 235: "nasi padang", 236: "nasi uduk",
 237: "babi guling", 238: "kaya toast", 239: "bak kut teh", 240: "curry puff",
 241: "chow mein", 242: "zha jiang mian", 243: "kung pao chicken",
 244: "crullers", 245: "eggplant with garlic sauce", 246: "three cup chicken",
 247: "bean curd family style", 248: "salt and pepper fried shrimp",
 249: "baked salmon", 250: "braised pork ball with napa cabbage",
 251: "winter melon soup", 252: "steamed spareribs", 253: "chinese pumpkin pie",
 254: "eight treasure rice", 255: "hot and sour soup",
}

# ---------------------------------------------------------------------
# Singleton State
# ---------------------------------------------------------------------

_session = None
_input_name = None
_input_shape = None
_model_type = None  # "onnx" or "ultralytics"


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------

def init_vision(model_name: str = "models/best_ep12.onnx") -> None:
    """
    Load the model into memory.
    Should be called once at startup.
    Automatically selects ONNX or Ultralytics based on extension.
    """
    global _session, _input_name, _input_shape, _model_type

    if _session is not None:
        logger.info("Vision engine already initialized.")
        return

    path = Path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    # -----------------------------------------------------------------
    # Ultralytics (.pt)
    # -----------------------------------------------------------------
    if path.suffix == ".pt":
        logger.info(f"Loading Ultralytics model: {path}")
        from ultralytics import YOLO

        _session = YOLO(str(path))
        _model_type = "ultralytics"
        logger.info("Ultralytics model loaded successfully.")
        return

    # -----------------------------------------------------------------
    # ONNX Runtime
    # -----------------------------------------------------------------
    logger.info(f"Loading ONNX model: {path}")

    import onnxruntime as ort

    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")

    providers.append("CPUExecutionProvider")

    _session = ort.InferenceSession(str(path), providers=providers)
    _input_name = _session.get_inputs()[0].name
    _input_shape = _session.get_inputs()[0].shape
    _model_type = "onnx"

    logger.info(f"ONNX model loaded. Input: {_input_name} {_input_shape}")
    logger.info(f"Providers: {providers}")


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def _preprocess(image_path: str, size: int = 640) -> np.ndarray:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size))

    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = np.expand_dims(arr, axis=0)

    return arr


# ---------------------------------------------------------------------
# Postprocessing (YOLO ONNX)
# ---------------------------------------------------------------------

def _postprocess(outputs: list, confidence: float = 0.25) -> dict[str, int]:
    """
    Parse YOLO ONNX output.
    Expected shape: [1, 4+num_classes, 8400]
    """

    predictions = outputs[0].squeeze(0)

    # Ensure shape [anchors, 4+num_classes]
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    class_scores = predictions[:, 4:]

    # Sigmoid activation
    class_scores = 1 / (1 + np.exp(-class_scores))

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]

    mask = scores >= confidence

    detections: dict[int, float] = {}

    for class_id, score in zip(class_ids[mask], scores[mask]):
        class_id = int(class_id)

        if class_id in FOOD_CLASSES:
            if class_id not in detections or score > detections[class_id]:
                detections[class_id] = float(score)

    # Keep top 5 detections
    top = sorted(detections.items(), key=lambda x: x[1], reverse=True)[:5]

    return {FOOD_CLASSES[cid]: 1 for cid, _ in top}


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def detect_food(image_path: str, confidence: float = 0.25) -> dict:
    """
    Run food detection.

    Returns:
    {
        "food_items": [...],
        "all_detections": {...}
    }
    """

    global _session

    # Lazy initialization
    if _session is None:
        logger.warning("Vision engine not initialized. Auto-loading default model.")
        init_vision()

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # -----------------------------------------------------------------
    # Ultralytics Path
    # -----------------------------------------------------------------
    if _model_type == "ultralytics":
        results = _session.predict(
            source=str(image_path),
            conf=confidence,
            imgsz=640,
            verbose=False,
        )

        all_detections = {}

        for box in results[0].boxes:
            label = results[0].names[int(box.cls[0])]
            all_detections[label] = all_detections.get(label, 0) + 1

        return {
            "food_items": list(all_detections.keys()),
            "all_detections": all_detections,
        }

    # -----------------------------------------------------------------
    # ONNX Path
    # -----------------------------------------------------------------
    img_array = _preprocess(str(image_path))
    outputs = _session.run(None, {_input_name: img_array})

    all_detections = _postprocess(outputs, confidence)

    return {
        "food_items": list(all_detections.keys()),
        "all_detections": all_detections,
    }