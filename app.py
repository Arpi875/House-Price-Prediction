import os
import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model_tuned.pkl")
MAP_PATH   = os.path.join(BASE_DIR, "location_map.pkl")

# ── Google Drive large file download ─────────────────────────────────────────
def download_gdrive_large(file_id: str, dest: str):
    print(f"[INFO] Downloading {dest} from Google Drive...")
    session = requests.Session()

    # Step 1: initial request
    response = session.get(
        "https://drive.google.com/uc",
        params={"export": "download", "id": file_id},
        stream=True
    )

    # Step 2: find confirmation token for large files
    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break

    # Also check response content for new-style token
    if not confirm_token:
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8", errors="ignore")
                if "confirm=" in line:
                    import re
                    match = re.search(r'confirm=([0-9A-Za-z_\-]+)', line)
                    if match:
                        confirm_token = match.group(1)
                        break

    # Step 3: download with confirm token
    if confirm_token:
        response = session.get(
            "https://drive.google.com/uc",
            params={"export": "download", "id": file_id, "confirm": confirm_token},
            stream=True
        )
    else:
        # Try alternate URL format
        response = session.get(
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
            stream=True
        )

    # Step 4: write file
    total = 0
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)
                total += len(chunk)

    size_mb = total / (1024 * 1024)
    print(f"[OK] Downloaded {dest} ({size_mb:.1f} MB)")
    return size_mb

# ── Download model if not present or too small ───────────────────────────────
MODEL_GDRIVE_ID = "1P-tRRDKTrcv95PSCFiyeuAgqMmfllopF"
MIN_SIZE_MB     = 50  # model should be > 50MB

def model_needs_download():
    if not os.path.exists(MODEL_PATH):
        return True
    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    if size_mb < MIN_SIZE_MB:
        print(f"[WARN] Model file too small ({size_mb:.1f} MB), re-downloading...")
        return True
    return False

if model_needs_download():
    try:
        download_gdrive_large(MODEL_GDRIVE_ID, MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")

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
    size_mb = os.path.getsize(MODEL_PATH) / (1024*1024) if os.path.exists(MODEL_PATH) else 0
    return jsonify({
        "status"       : "ok",
        "model_loaded" : model is not None,
        "model_size_mb": round(size_mb, 1),
        "locations"    : len(location_map)
    })

@app.route("/locations")
def locations():
    return jsonify({"locations": sorted(location_map.keys())})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please try again in a moment."}), 500
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
