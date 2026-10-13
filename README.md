# 🏠 House Price Predictor

Predict the price of a residential property in India (in **INR Lakhs**) from details such as city, locality type, area, BHK, building age and distance to the city centre.

The model is a **linear regression written from scratch in NumPy**, trained with **batch gradient descent** and **L1 (lasso) regularisation**. The trained model is served through two interfaces: a **Streamlit** app and a **Flask** REST API with an HTML front-end that also explains each prediction.

---

## ✨ Features

- Gradient descent, L2 and L1 penalties implemented by hand (no `sklearn` model classes)
- Data cleaning, log transforms, one-hot encoding and standardisation
- One pickle file holds everything needed for prediction (weights, bias, encoder, scaler, column order)
- Streamlit app for quick predictions
- Flask API (`/api/predict`) that returns the price plus a plain-English explanation of which features pushed the price up or down

---

## 🗂️ Project Structure

```
.
├── model.py               # Exploratory notebook export (Colab): EDA, plain GD, L2, L1, plots
├── train_and_save.py      # Trains the L1 model and writes model_artifacts.pkl
├── app.py                 # Streamlit web app
├── server.py              # Flask backend (serves templates/index.html + REST API)
├── templates/index.html   # Front-end for the Flask app
├── requirements.txt
└── .gitignore             # train.csv and model_artifacts.pkl are not committed
```

> `train.csv` and `model_artifacts.pkl` are git-ignored. Place `train.csv` in the project root and generate the pickle with `train_and_save.py`.

---

## ⚙️ Setup

```bash
git clone <your-repo-url>
cd <your-repo-folder>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install streamlit           # needed for app.py
```

---

## 🚀 Usage

### 1. Train the model

Put `train.csv` in the project folder, then run:

```bash
python train_and_save.py
```

This prints the test-set **MAE** and **R²** and saves `model_artifacts.pkl`.

### 2a. Run the Streamlit app

```bash
streamlit run app.py
```

### 2b. Or run the Flask app

```bash
python server.py
```

Then open <http://127.0.0.1:5000>.

#### API example

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "super_area": 1200, "carpet_area": 900, "bhk": 2, "bathrooms": 2,
    "floor_number": 3, "total_floors": 10, "age": 5,
    "dist_metro": 2, "dist_center": 5,
    "city": "Hyderabad", "locality_type": "Mid-Range",
    "property_type": "Apartment", "furnishing": "Semi-Furnished"
  }'
```

The response contains `price_lakhs`, `price_rupees`, `price_crores` and an `explanation` object (category contributions, top positive and negative drivers, and a short summary).

---

## 🧠 How It Works

| Step | What happens |
|------|--------------|
| **Cleaning** | Drop `Property_ID`; remove listings with super area < 200 sq ft or < 100 sq ft per BHK |
| **Transforms** | `log1p` on `Super_Area_SqFt` and `Carpet_Area_SqFt`; `log1p` on the target price |
| **Split** | 80 / 20 train–test, `random_state=42` |
| **Encoding** | One-hot: `City`, `Locality_Type`, `Property_Type`, `Furnishing_Status` (fit on train only) |
| **Scaling** | `StandardScaler` on 9 numeric columns (fit on train only) |
| **Training** | Batch gradient descent, `lr = 0.1`, 1000 iterations, L1 penalty `λ = 0.01` |
| **Prediction** | `price = expm1(w · x + b)`, clipped at 0 |

The notebook (`model.py`) also compares plain gradient descent, L2 (`λ = 0.001`) and L1 (`λ = 0.01`).

### Features used (32 after encoding)

- **Numeric:** BHK, bathrooms, log super area, log carpet area, floor number, total floors, age, distance to metro, distance to city centre
- **Amenity columns from the dataset:** Parking, Lift_Available, Gated_Community
- **Categorical (one-hot):** City (10), Locality type (3), Property type (4), Furnishing status (3)

---

## 📊 What the Model Learned

Taken from the saved weights (log-price space):

- **Location dominates.** Relative to Hyderabad, Mumbai is about **2.5×** and Delhi about **1.8×** as expensive; Jaipur is about **0.56×**.
- **Locality type matters.** Premium is about **1.9×** and Mid-Range about **1.3×** an Affordable locality.
- **Area is close to proportional.** The log-area coefficient implies price scales roughly with area^0.94.
- **Older and farther is cheaper.** Each standard deviation of age (about 11.5 years) is worth roughly −10%; each standard deviation of distance to the city centre (about 3.9 km) roughly −9%.
- **L1 sparsity:** 10 of the 32 weights are below 10⁻³ in magnitude, including BHK, bathrooms, distance to metro and the property-type indicators.
- Furnishing status has almost no effect (the three weights differ by under 0.01).

### Test performance

Run `python train_and_save.py` to print your own numbers, then record them here:

| Model | MAE (Lakhs) | R² |
|-------|-------------|----|
| Plain gradient descent | _fill in_ | _fill in_ |
| L2 (λ = 0.001) | _fill in_ | _fill in_ |
| **L1 (λ = 0.01), deployed** | _fill in_ | _fill in_ |

---
## 🔭 Future Work

- Add the three amenity inputs to both interfaces
- Try elastic net, random forest and gradient boosting as comparison models
- K-fold cross-validation and tuning of `λ` and the learning rate
- Prediction intervals instead of a single number
- Deploy to Streamlit Community Cloud or a container

---

## 🛠️ Tech Stack

Python · NumPy · pandas · scikit-learn (preprocessing and metrics only) · Matplotlib · Seaborn · Streamlit · Flask

---
