import streamlit as st
import pandas as pd
import pickle

def load_models():
    clf = pickle.load(open("knn_classifier.pkl", "rb"))
    reg = pickle.load(open("knn_regressor.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return clf, reg, scaler

def get_input():
    st.write("### 🍇 Enter Wine Features")
    features = {
        "fixed acidity": st.number_input("Fixed Acidity", value=7.4),
        "volatile acidity": st.number_input("Volatile Acidity", value=0.7),
        "citric acid": st.number_input("Citric Acid", value=0.0),
        "residual sugar": st.number_input("Residual Sugar", value=1.9),
        "chlorides": st.number_input("Chlorides", value=0.076),
        "free sulfur dioxide": st.number_input("Free Sulfur Dioxide", value=11.0),
        "total sulfur dioxide": st.number_input("Total Sulfur Dioxide", value=34.0),
        "density": st.number_input("Density", value=0.9978),
        "pH": st.number_input("pH", value=3.51),
        "sulphates": st.number_input("Sulphates", value=0.56),
        "alcohol": st.number_input("Alcohol", value=9.4)
    }
    return pd.DataFrame([features])

def run_app():
    st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷")
    st.title("🍷 Unified Wine Quality Predictor")

    clf, reg, scaler = load_models()
    df = get_input()
    scaled = scaler.transform(df)

    tab1, tab2 = st.tabs(["Classification", "Regression"])

    with tab1:
        st.subheader("🔍 Classification")
        pred_class = clf.predict(scaled)[0]
        st.success(f"Predicted Quality Class: **{pred_class}**")

    with tab2:
        st.subheader("📈 Regression")
        pred_score = reg.predict(scaled)[0]
        st.success(f"Predicted Quality Score: **{round(pred_score, 2)}**")

if __name__ == "__main__":
    run_app()