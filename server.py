"""
server.py  –  Flask backend for the House Price Predictor
Run with:  python server.py
Then open: http://127.0.0.1:5000
"""

import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Load model artifacts ──────────────────────────────────────────────────────
try:
    with open("model_artifacts.pkl", "rb") as f:
        arts = pickle.load(f)
except FileNotFoundError:
    raise SystemExit("❌  model_artifacts.pkl not found. Run train_and_save.py first.")

w            = arts["w"]
b            = arts["b"]
encoder      = arts["encoder"]
scaler       = arts["scaler"]
onehot_cols  = arts["onehot_cols"]
numeric_cols = arts["numeric_cols"]
feature_cols = arts["feature_cols"]

cat_options = {
    col: cats.tolist()
    for col, cats in zip(onehot_cols, encoder.categories_)
}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/categories")
def categories():
    return jsonify(cat_options)


@app.route("/api/predict", methods=["POST"])
def predict():
    d = request.get_json()

    try:
        super_area  = float(d["super_area"])
        carpet_area = float(d["carpet_area"])
        log_super   = np.log1p(super_area)
        log_carpet  = np.log1p(carpet_area)

        # Build numeric row
        num_df = pd.DataFrame([[
            float(d["bhk"]),          float(d["bathrooms"]),
            log_super,                log_carpet,
            float(d["floor_number"]), float(d["total_floors"]),
            float(d["age"]),          float(d["dist_metro"]),
            float(d["dist_center"])
        ]], columns=numeric_cols)

        num_scaled = scaler.transform(num_df)
        num_df     = pd.DataFrame(num_scaled, columns=numeric_cols)

        # Build categorical row
        cat_df_raw = pd.DataFrame([[
            d["city"], d["locality_type"], d["property_type"], d["furnishing"]
        ]], columns=onehot_cols)

        cat_enc  = encoder.transform(cat_df_raw)
        enc_cols = encoder.get_feature_names_out(onehot_cols)
        cat_df   = pd.DataFrame(cat_enc, columns=enc_cols)

        # Combine & align
        row     = pd.concat([num_df, cat_df], axis=1).reindex(columns=feature_cols, fill_value=0)
        row_arr = np.asarray(row, dtype=float).flatten()

        log_price = float(row_arr @ w + b)
        price     = float(max(np.expm1(log_price), 0.0))

        # ── Explanation ───────────────────────────────────────────────────────
        contributions = row_arr * w          # each feature's log-space contribution

        # Friendly display names for numeric features
        FRIENDLY = {
            "BHK":                        f"BHK ({d['bhk']})",
            "Bathrooms":                  f"Bathrooms ({d['bathrooms']})",
            "Log_Super_Area":             f"Super area ({int(super_area):,} sq ft)",
            "Log_Carpet_Area":            f"Carpet area ({int(carpet_area):,} sq ft)",
            "Floor_Number":               f"Floor number ({d['floor_number']})",
            "Total_Floors":               f"Building height ({d['total_floors']} floors)",
            "Age_of_Property":            f"Property age ({d['age']} yrs)",
            "Distance_to_Metro_km":       f"Distance to metro ({d['dist_metro']} km)",
            "Distance_to_City_Center_km": f"Distance to city center ({d['dist_center']} km)",
        }

        # Sum contributions per logical group
        groups = {}
        for i, col in enumerate(feature_cols):
            if   col.startswith("City_"):              key = "city"
            elif col.startswith("Locality_Type_"):     key = "locality"
            elif col.startswith("Property_Type_"):     key = "property"
            elif col.startswith("Furnishing_Status_"): key = "furnishing"
            else:                                       key = col
            groups[key] = groups.get(key, 0) + contributions[i]

        FRIENDLY_GROUPS = {
            "city":       f"City: {d['city']}",
            "locality":   f"Locality: {d['locality_type']}",
            "property":   f"Type: {d['property_type']}",
            "furnishing": f"Furnishing: {d['furnishing']}",
        }
        FRIENDLY_GROUPS.update(FRIENDLY)

        # Per-item list sorted by impact
        items = sorted([
            {
                "name":    FRIENDLY_GROUPS.get(k, k),
                "contrib": round(float(v), 4),
                "pct":     round(float((np.expm1(v)) * 100), 1),
            }
            for k, v in groups.items()
        ], key=lambda x: x["contrib"], reverse=True)

        # Category-level totals for bar chart
        loc_keys  = {"city", "locality", "property", "furnishing"}
        size_keys = {"BHK", "Bathrooms", "Log_Super_Area", "Log_Carpet_Area"}

        loc_sum  = sum(groups.get(k, 0) for k in loc_keys)
        size_sum = sum(groups.get(k, 0) for k in size_keys)
        bldg_sum = sum(v for k, v in groups.items()
                       if k not in loc_keys and k not in size_keys)

        total_abs = abs(loc_sum) + abs(size_sum) + abs(bldg_sum) or 1
        categories = [
            {"label": "Location", "icon": "📍",
             "pct": round(abs(loc_sum)  / total_abs * 100), "positive": bool(loc_sum  >= 0)},
            {"label": "Size",     "icon": "📐",
             "pct": round(abs(size_sum) / total_abs * 100), "positive": bool(size_sum >= 0)},
            {"label": "Building", "icon": "🏢",
             "pct": round(abs(bldg_sum) / total_abs * 100), "positive": bool(bldg_sum >= 0)},
        ]

        top_positive = [x for x in items if x["contrib"] > 0][:3]
        top_negative = [x for x in items if x["contrib"] < 0][:3]

        # Plain-English summary
        dominant = max(
            {"Location": abs(loc_sum), "Size": abs(size_sum), "Building": abs(bldg_sum)},
            key=lambda k: {"Location": abs(loc_sum), "Size": abs(size_sum), "Building": abs(bldg_sum)}[k]
        )
        summary = (f"This {d['bhk']}-BHK {d['property_type'].lower()} in {d['city']} is "
                   f"estimated at ₹{round(price, 1):,} Lakhs. ")
        if dominant == "Location":
            summary += (f"Location is the biggest driver — {d['locality_type']} areas "
                        f"in {d['city']} carry the most weight in the model. ")
        elif dominant == "Size":
            summary += (f"The property's size ({int(super_area):,} sq ft, {d['bhk']} BHK) "
                        f"is the primary value factor. ")
        else:
            summary += ("Building characteristics like age, floor level, and proximity "
                        "to amenities dominate the estimate. ")
        if top_positive:
            summary += f"The strongest upward driver is {top_positive[0]['name'].lower()}."
        if top_negative:
            summary += f" {top_negative[0]['name']} has a moderating downward effect."

        return jsonify({
            "price_lakhs":  round(price, 2),
            "price_rupees": round(price * 1_00_000),
            "price_crores": round(price / 100, 3),
            "explanation": {
                "categories":    categories,
                "top_positive":  top_positive,
                "top_negative":  top_negative,
                "summary":       summary,
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == "__main__":
    app.run(debug=True)
