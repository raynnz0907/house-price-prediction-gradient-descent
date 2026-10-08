"""
app.py  –  Streamlit House Price Predictor
Run with:  streamlit run app.py
(train_and_save.py must have been run first to generate model_artifacts.pkl)
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

try:
    arts = load_artifacts()
except FileNotFoundError:
    st.error(
        "⚠️ `model_artifacts.pkl` not found.\n\n"
        "Please run `python train_and_save.py` first to train and save the model."
    )
    st.stop()

w            = arts["w"]
b            = arts["b"]
encoder      = arts["encoder"]
scaler       = arts["scaler"]
onehot_cols  = arts["onehot_cols"]
numeric_cols = arts["numeric_cols"]
feature_cols = arts["feature_cols"]

# Pull category lists directly from the fitted encoder
cat_options = {col: enc_cats.tolist() for col, enc_cats in zip(onehot_cols, encoder.categories_)}

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏠 House Price Predictor")
st.markdown("Fill in the property details below and click **Predict Price**.")

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📐 Size & Layout")
    super_area   = st.number_input("Super Area (sq ft)",   min_value=200,  max_value=20000, value=1200, step=50)
    carpet_area  = st.number_input("Carpet Area (sq ft)",  min_value=100,  max_value=15000, value=900,  step=50)
    bhk          = st.number_input("BHK",                  min_value=1,    max_value=10,    value=2,    step=1)
    bathrooms    = st.number_input("Bathrooms",             min_value=1,    max_value=10,    value=2,    step=1)

    st.subheader("🏢 Building Details")
    floor_number = st.number_input("Floor Number",          min_value=0,    max_value=100,   value=3,    step=1)
    total_floors = st.number_input("Total Floors",          min_value=1,    max_value=100,   value=10,   step=1)
    age          = st.number_input("Age of Property (yrs)", min_value=0,    max_value=100,   value=5,    step=1)

with col2:
    st.subheader("📍 Location")
    city          = st.selectbox("City",          cat_options["City"])
    locality_type = st.selectbox("Locality Type", cat_options["Locality_Type"])
    dist_metro    = st.number_input("Distance to Metro (km)",       min_value=0.0, max_value=50.0, value=2.0, step=0.1)
    dist_center   = st.number_input("Distance to City Center (km)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)

    st.subheader("🏠 Property Info")
    property_type    = st.selectbox("Property Type",     cat_options["Property_Type"])
    furnishing_status = st.selectbox("Furnishing Status", cat_options["Furnishing_Status"])

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(super_area, carpet_area, bhk, bathrooms, floor_number,
            total_floors, age, dist_metro, dist_center,
            city, locality_type, property_type, furnishing_status):

    # Log-transform area (same as training)
    log_super  = np.log1p(super_area)
    log_carpet = np.log1p(carpet_area)

    # Build a single-row DataFrame with numeric features
    num_data = pd.DataFrame([[bhk, bathrooms, log_super, log_carpet,
                              floor_number, total_floors, age,
                              dist_metro, dist_center]],
                            columns=numeric_cols)

    # Scale numeric features
    num_scaled = scaler.transform(num_data)
    num_df     = pd.DataFrame(num_scaled, columns=numeric_cols)

    # One-hot encode categorical features
    cat_data = pd.DataFrame([[city, locality_type, property_type, furnishing_status]],
                            columns=onehot_cols)
    cat_enc  = encoder.transform(cat_data)
    enc_names = encoder.get_feature_names_out(onehot_cols)
    cat_df   = pd.DataFrame(cat_enc, columns=enc_names)

    # Combine and align columns with training order
    row = pd.concat([num_df, cat_df], axis=1)
    row = row.reindex(columns=feature_cols, fill_value=0)

    # Predict (model was trained on log1p(price))
    log_price = float(np.asarray(row, dtype=float).flatten() @ w + b)
    price     = np.expm1(log_price)
    return max(price, 0.0)   # clip negatives just in case


if st.button("🔍 Predict Price", use_container_width=True, type="primary"):
    price = predict(
        super_area, carpet_area, bhk, bathrooms,
        floor_number, total_floors, age, dist_metro, dist_center,
        city, locality_type, property_type, furnishing_status,
    )

    st.success(f"### 💰 Estimated Price: ₹ {price:,.2f} Lakhs")
    st.caption(f"≈ ₹ {price * 1_00_000:,.0f}  |  ≈ ₹ {price / 100:.2f} Crore")

    # Quick summary card
    with st.expander("📋 Input Summary"):
        st.json({
            "City":             city,
            "Locality Type":    locality_type,
            "Property Type":    property_type,
            "Furnishing":       furnishing_status,
            "BHK":              bhk,
            "Bathrooms":        bathrooms,
            "Super Area (sqft)": super_area,
            "Carpet Area (sqft)": carpet_area,
            "Floor":            floor_number,
            "Total Floors":     total_floors,
            "Age (yrs)":        age,
            "Metro Distance":   dist_metro,
            "City Center Dist": dist_center,
        })
