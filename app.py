import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_tuned.pkl")
MAP_PATH   = os.path.join(BASE_DIR, "location_map.pkl")

# ── Load model + location map ─────────────────────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Model loaded successfully")
except Exception as e:
    model = None
    print(f"[WARN] Model load failed: {e}")

try:
    location_map = joblib.load(MAP_PATH)
    print(f"[OK] Location map loaded: {len(location_map)} locations")
except Exception as e:
    location_map = {}
    print(f"[WARN] Location map load failed: {e}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def format_price(price_lac: float) -> str:
    if price_lac >= 100:
        return f"₹{price_lac/100:.2f} Crore"
    return f"₹{price_lac:.2f} Lac"

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/health")
def health():
    return jsonify({
        "status"      : "ok",
        "model_loaded": model is not None,
        "locations"   : len(location_map)
    })

@app.route("/locations")
def locations():
    return jsonify({"locations": sorted(location_map.keys())})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500
    try:
        data = request.get_json(force=True)

        bhk            = float(data.get("bhk", 2))
        carpet_sqft    = float(data.get("carpet_sqft", 0))
        floor_num      = float(data.get("floor_num", 1))
        total_floors   = float(data.get("total_floors", 1))
        location       = str(data.get("location", "")).strip().lower()
        is_ready       = int(data.get("is_ready", 1))
        is_resale      = int(data.get("is_resale", 1))
        bathroom       = float(data.get("bathroom", 2))
        price_per_sqft = float(data.get("price_per_sqft", 0))

        floor_ratio      = floor_num / total_floors if total_floors > 0 else 0
        location_encoded = location_map.get(location, 1)

        features = ["BHK","Carpet_sqft","Floor_num","Total_floors",
                    "Floor_ratio","location_encoded","is_ready",
                    "is_resale","Bathroom","price_per_sqft"]

        row = pd.DataFrame([{
            "BHK"             : bhk,
            "Carpet_sqft"     : carpet_sqft,
            "Floor_num"       : floor_num,
            "Total_floors"    : total_floors,
            "Floor_ratio"     : floor_ratio,
            "location_encoded": location_encoded,
            "is_ready"        : is_ready,
            "is_resale"       : is_resale,
            "Bathroom"        : bathroom,
            "price_per_sqft"  : price_per_sqft
        }])[features]

        log_pred  = model.predict(row)[0]
        price_lac = float(np.expm1(log_pred))

        return jsonify({
            "status"        : "success",
            "price_lac"     : round(price_lac, 2),
            "price_display" : format_price(price_lac),
            "location_known": location in location_map,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
