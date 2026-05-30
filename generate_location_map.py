"""
Run this script ONCE on your local machine to generate location_map.pkl
from your original dataset.

Usage:
    python generate_location_map.py

It will create location_map.pkl in the same folder.
Place both best_model_tuned.pkl and location_map.pkl next to app.py.
"""

import pandas as pd
import joblib
import os

# ── Apna CSV path yahan set karein ───────────────────────────────────────────
CSV_PATH = "house_prices.csv"   # same folder mein ho toh bas naam likhein

df = pd.read_csv(CSV_PATH)

# Exact same logic as notebook Experiment 7
df["location"] = df["location"].str.strip().str.lower()
location_map   = df["location"].value_counts().to_dict()

# Save
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "location_map.pkl")
joblib.dump(location_map, OUT)

print(f"[OK] location_map.pkl saved with {len(location_map)} locations.")
print("Top 10 locations:")
for loc, count in list(location_map.items())[:10]:
    print(f"  {loc:<25} → {count}")
