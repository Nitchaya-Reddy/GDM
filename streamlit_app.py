import pickle
import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load pre-trained models and scaler
ensemble_model = pickle.load(open('sources/ensemble_model.pkl', 'rb'))
cnn_model = load_model('sources/cnn_model.h5')
scaler = joblib.load('sources/scaler.pkl')

def predict_combined(input_data, ensemble_weight=0.3, cnn_weight=0.7):
    """
    Predict GDM using Ensemble and CNN models with a weighted combination.

    Parameters:
        input_data (numpy array): Preprocessed input data (scaled).
        ensemble_weight (float): Weight for the ensemble model's prediction.
        cnn_weight (float): Weight for the CNN model's prediction.

    Returns:
        dict: Dictionary containing individual model probabilities and final combined output.
    """
    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    # Predict with ensemble model
    ensemble_pred_proba = ensemble_model.predict_proba(input_data_scaled)[0][1]

    # Prepare input for CNN model
    cnn_input = input_data_scaled[..., np.newaxis]  # Add a feature dimension
    cnn_pred_proba = cnn_model.predict(cnn_input, verbose=1)[0][0]

    # Combine predictions using weights
    combined_proba = (ensemble_weight * ensemble_pred_proba) + (cnn_weight * cnn_pred_proba)
    final_prediction = "GDM" if combined_proba > 0.5 else "Non-GDM"

    return {
        "ensemble_proba": ensemble_pred_proba,
        "cnn_proba": cnn_pred_proba,
        "combined_proba": combined_proba,
        "final_prediction": final_prediction
    }

# Set page layout
st.set_page_config(page_title="GDM Prediction", layout="wide")

# Streamlit app
st.title("Gestational Diabetes Mellitus (GDM) Prediction")
st.write("This app predicts the likelihood of GDM using both CNN and Ensemble models with a combined prediction.")

# User Input Form
st.header("Enter Patient Details")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
gestation = st.number_input("Gestation in Weeks", min_value=0, max_value=50, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
hdl = st.number_input("HDL Cholesterol", min_value=0.0, max_value=100.0, value=50.0)
family_history = st.selectbox("Family History of Diabetes (Yes=1, No=0)", [0, 1])
pcos = st.selectbox("PCOS (Yes=1, No=0)", [0, 1])
dia_bp = st.number_input("Diastolic Blood Pressure", min_value=0.0, max_value=200.0, value=80.0)
ogtt = st.number_input("OGTT (mg/dL)", min_value=0.0, max_value=500.0, value=140.0)
hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, value=12.0)
prediabetes = st.selectbox("History of Prediabetes (Yes=1, No=0)", [0, 1])

# Prediction Button
if st.button("Predict GDM"):
    # Gather input data
    input_data = np.array([[age, gestation, bmi, hdl, family_history, pcos, dia_bp, ogtt, hemoglobin, prediabetes]])

    # Call backend function to predict
    prediction_result = predict_combined(input_data)

    # Display results
    st.subheader("Prediction Results")
    st.write(f"Ensemble Model Probability: {prediction_result['ensemble_proba']:.2f}")
    st.write(f"CNN Model Probability: {prediction_result['cnn_proba']:.2f}")
    st.write(f"Combined Probability: {prediction_result['combined_proba']:.2f}")
    st.write(f"Final Prediction: {prediction_result['final_prediction']}")
    st.subheader("Probability Distribution")
    

# Display disclaimer
st.subheader("Disclaimer")
st.write("This app is for educational purposes only. The results are not intended to be used as medical advice.")
