# app.py - Bengaluru House Price Prediction with 2025 Inflation Adjustment
# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="house",
    layout="centered"
)

MODEL_PATH = "Ridgemodel.pkl"
DATA_URL = "https://raw.githubusercontent.com/krishnaik06/Advanced-House-Price-Prediction/master/Bengaluru_House_Data.csv"

# Premium & High-Growth Areas in Bengaluru (Nov 2025)
PREMIUM_AREAS = {
    'Indiranagar', 'Koramangala', 'Malleshwaram', 'Sadashivanagar',
    'Jayanagar', 'Richards Town', 'Lavelle Road', 'MG Road', 'Vasanth Nagar'
}
HIGH_GROWTH_AREAS = {
    'Whitefield', 'Sarjapur Road', 'Electronic City', 'Bellandur',
    'Marathahalli', 'HSR Layout', 'Bannerghatta Road', 'Hebbal', 'Yelahanka'
}

@st.cache_resource
def load_model_and_locations():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        st.warning("Model not found. Training now (first run only)...")
        model = train_and_save_model()
    
    # Extract locations from the OneHotEncoder inside the pipeline
    ohe = model.named_steps['columntransformer'].named_transformers_['onehotencoder']
    locations = [col.split('_')[-1] for col in ohe.get_feature_names_out()]
    locations = sorted(set(locations))
    return model, locations

def train_and_save_model():
    df = pd.read_csv(DATA_URL)

    def convert_sqft(x):
        x = str(x).strip()
        if '-' in x:
            tokens = x.split('-')
            if len(tokens) == 2:
                return (float(tokens[0]) + float(tokens[1])) / 2
        tokens = re.findall(r'\d+\.?\d*', x)
        if not tokens:
            return np.nan
        num = float(tokens[0])
        if 'Sq. Meter' in x: return num * 10.7639
        elif 'Sq. Yards' in x: return num * 9
        elif 'Acres' in x: return num * 43560
        elif 'Cents' in x: return num * 435.56
        elif 'Guntha' in x: return num * 1089
        elif 'Grounds' in x: return num * 2400
        elif 'Perch' in x: return num * 272.25
        else: return num

    df['total_sqft_cleaned'] = df['total_sqft'].apply(convert_sqft)
    df = df.dropna(subset=['total_sqft_cleaned'])
    df['bhk'] = df['size'].fillna('2 BHK').str.extract('(\d+)').astype(int)
    df['location'] = df['location'].fillna('Whitefield').str.strip()

    # Outlier removal (same as notebook)
    def remove_pps_outliers(df):
        df_out = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st_d = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft > (m - st_d)) & (subdf.price_per_sqft <= (m + st_d))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out

    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft_cleaned']
    df = remove_pps_outliers(df)

    def remove_bhk_outliers(df):
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.price_per_sqft),
                    'std': np.std(bhk_df.price_per_sqft),
                    'count': bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk-1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
        return df.drop(exclude_indices, axis='index')

    df = remove_bhk_outliers(df)
    df['location'] = df['location'].apply(lambda x: x if df['location'].value_counts()[x] > 10 else 'Other')

    X = df[['location', 'total_sqft_cleaned', 'bhk']]
    y = df['price']

    preprocessor = ColumnTransformer([
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['location'])
    ], remainder='passthrough')

    pipeline = Pipeline([
        ('columntransformer', preprocessor),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])

    pipeline.fit(X, y)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    
    st.success("Model trained and saved!")
    return pipeline

# Load model
model, locations = load_model_and_locations()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Bengaluru House Price Predictor")
st.markdown("### Accurate predictions with **2025 inflation adjustment**")

col1, col2 = st.columns(2)
with col1:
    location = st.selectbox("Location", options=sorted(locations), index=locations.index("Whitefield") if "Whitefield" in locations else 0)
with col2:
    bhk = st.selectbox("BHK", options=list(range(1, 11)), index=2)

sqft = st.slider("Total Area (sqft)", min_value=300, max_value=20000, value=1250, step=50)

if st.button("Predict Price", type="primary"):
    with st.spinner("Calculating..."):
        input_df = pd.DataFrame({
            'location': [location],
            'total_sqft_cleaned': [sqft],
            'bhk': [bhk]
        })

        base_price = model.predict(input_df)[0]

        # 2025 Inflation Multiplier
        if location in PREMIUM_AREAS:
            multiplier = 3.1
            tier = "Ultra-Premium"
        elif location in HIGH_GROWTH_AREAS:
            multiplier = 2.75
            tier = "High-Growth"
        elif location == "Other":
            multiplier = 2.3
            tier = "Average"
        else:
            multiplier = 2.5
            tier = "Good"

        final_price = base_price * multiplier

        st.success("Prediction Complete!")
        st.metric("Base Model Price (2020 data)", f"₹{base_price:.2f} Lakh")
        st.metric(f"Estimated Price in **Nov 2025**", f"₹{final_price:.2f} Lakh", delta=f"+{multiplier:.1f}x since 2020")

        col3, col4 = st.columns(2)
        with col3:
            st.info(f"**Location Tier**: {tier}")
        with col4:
            st.info(f"**Inflation Multiplier**: ×{multiplier:.1f}")

st.markdown("---")
st.caption("Model trained on 13K+ Bengaluru properties | Adjusted for 2025 market trends | Made with Streamlit")
