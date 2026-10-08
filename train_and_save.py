"""
train_and_save.py
Run this ONCE to train the L1 model and save artifacts for the Streamlit app.

Usage:
    python train_and_save.py
    (make sure train.csv is in the same folder, or change CSV_PATH below)
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH      = "train.csv"   # <-- change if your CSV is elsewhere
ARTIFACT_PATH = "model_artifacts.pkl"
# ─────────────────────────────────────────────────────────────────────────────

# 1. Load & clean
df = pd.read_csv(CSV_PATH)
df.drop(["Property_ID"], axis=1, inplace=True)

bad_area = (df["Super_Area_SqFt"] / df["BHK"] < 100) | (df["Super_Area_SqFt"] < 200)
df = df[~bad_area].copy()

# 2. Log-transform area columns
df["Log_Super_Area"]  = np.log1p(df["Super_Area_SqFt"])
df["Log_Carpet_Area"] = np.log1p(df["Carpet_Area_SqFt"])

# 3. Split
X = df.drop(["Price_INR_Lakhs", "Super_Area_SqFt", "Carpet_Area_SqFt"], axis=1)
y = df["Price_INR_Lakhs"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. One-hot encoding
onehot_cols = ["City", "Locality_Type", "Property_Type", "Furnishing_Status"]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_train_enc = encoder.fit_transform(X_train[onehot_cols])
X_test_enc  = encoder.transform(X_test[onehot_cols])

enc_names    = encoder.get_feature_names_out(onehot_cols)
X_train_enc  = pd.DataFrame(X_train_enc, columns=enc_names, index=X_train.index)
X_test_enc   = pd.DataFrame(X_test_enc,  columns=enc_names, index=X_test.index)

X_train_final = pd.concat([X_train.drop(onehot_cols, axis=1), X_train_enc], axis=1)
X_test_final  = pd.concat([X_test.drop(onehot_cols, axis=1),  X_test_enc],  axis=1)

# 5. Standard scaling
numeric_cols = ["BHK", "Bathrooms", "Log_Super_Area", "Log_Carpet_Area",
                "Floor_Number", "Total_Floors", "Age_of_Property",
                "Distance_to_Metro_km", "Distance_to_City_Center_km"]

scaler = StandardScaler()
X_train_final[numeric_cols] = scaler.fit_transform(X_train_final[numeric_cols])
X_test_final[numeric_cols]  = scaler.transform(X_test_final[numeric_cols])

# Save column order (critical for correct prediction later)
feature_cols = X_train_final.columns.tolist()

# 6. Train L1 (Lasso) model via gradient descent
X_arr = np.asarray(X_train_final, dtype=float)
y_arr = np.asarray(np.log1p(y_train), dtype=float)

n, d = X_arr.shape
w = np.zeros(d)
b = 0.0
lr = 0.1
n_iters = 1000
lam = 0.01

for i in range(n_iters):
    y_pred  = X_arr @ w + b
    error   = y_pred - y_arr
    grad_w  = (2 / n) * (X_arr.T @ error) + lam * np.sign(w)
    grad_b  = (2 / n) * np.sum(error)
    w -= lr * grad_w
    b -= lr * grad_b

# 7. Evaluate
from sklearn.metrics import mean_absolute_error, r2_score
pred_lakhs = np.expm1(np.asarray(X_test_final, dtype=float) @ w + b)
print(f"MAE : {mean_absolute_error(y_test, pred_lakhs):.2f} Lakhs")
print(f"R²  : {r2_score(y_test, pred_lakhs):.4f}")

# 8. Save all artifacts
artifacts = {
    "w":            w,
    "b":            b,
    "encoder":      encoder,
    "scaler":       scaler,
    "onehot_cols":  onehot_cols,
    "numeric_cols": numeric_cols,
    "feature_cols": feature_cols,
}

with open(ARTIFACT_PATH, "wb") as f:
    pickle.dump(artifacts, f)

print(f"\n✅  Artifacts saved to '{ARTIFACT_PATH}'")
print("   Now run:  streamlit run app.py")
