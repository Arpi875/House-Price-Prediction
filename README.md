# 🏠 House Price Predictor — Complete Setup Guide

## 📁 Project Structure
```
house_price_project/
├── app.py                    ← Flask backend (API)
├── requirements.txt          ← Python packages
├── render.yaml               ← Render.com deploy config
├── generate_location_map.py  ← Run once to create location_map.pkl
├── best_model_tuned.pkl      ← ← YOUR TRAINED MODEL (aapko add karna hai)
├── location_map.pkl          ← ← AUTO GENERATED (niche steps follow karein)
└── static/
    └── index.html            ← Frontend website
```

---

## ✅ STEP 1 — Model files taiyaar karein (LOCAL)

Kaggle notebook mein yeh 2 cells run karein:

### Cell A — Model save karo
```python
import joblib
joblib.dump(best_rf, 'best_model_tuned.pkl')
print("Model saved!")
```

### Cell B — Location map save karo
```python
import joblib
location_map = df['location'].value_counts().to_dict()
joblib.dump(location_map, 'location_map.pkl')
print(f"Location map saved! {len(location_map)} locations")
```

Kaggle Output panel se dono files download karein.

---

## ✅ STEP 2 — Local test karein

```bash
# 1. Project folder mein jaao
cd house_price_project

# 2. best_model_tuned.pkl aur location_map.pkl yahan rakhein

# 3. Packages install karein
pip install -r requirements.txt

# 4. Server start karein
python app.py

# 5. Browser mein kholein
# http://localhost:5000
```

---

## ✅ STEP 3 — GitHub par upload karein

```bash
git init
git add .
git commit -m "House price predictor"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/house-price-predictor.git
git push -u origin main
```

⚠️ IMPORTANT: `best_model_tuned.pkl` aur `location_map.pkl` bhi push karein.

---

## ✅ STEP 4 — Render.com par FREE deploy karein

1. https://render.com par jaao → Sign up (GitHub se)
2. **New** → **Web Service** → GitHub repo select karein
3. Settings:
   - **Name**: house-price-predictor
   - **Branch**: main
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free
4. **Create Web Service** click karein
5. 3-5 minute mein deploy ho jaayega
6. Aapko ek URL milega: `https://house-price-predictor.onrender.com`

---

## 🧪 API Test (Terminal se)

```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bhk": 2,
    "carpet_sqft": 650,
    "floor_num": 10,
    "total_floors": 22,
    "location": "thane",
    "is_ready": 1,
    "is_resale": 1,
    "bathroom": 2,
    "price_per_sqft": 13000
  }'
```

Expected response:
```json
{
  "status": "success",
  "price_lac": 84.5,
  "price_display": "₹84.50 Lac",
  "location_known": true,
  "floor_ratio": 0.455
}
```

---

## ⚠️ Common Issues

| Problem | Solution |
|---------|----------|
| `best_model_tuned.pkl not found` | Kaggle se download karke project folder mein rakhein |
| `location_known: false` | Location name exactly match karna chahiye (e.g. "thane" not "Thane") |
| Render deploy fail | `requirements.txt` mein versions check karein |
| Slow on first request | Free tier par server sleeps — 30 sec wait karein |
