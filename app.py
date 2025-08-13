import streamlit as st
import joblib
import pickle
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="ML Model Predictions",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def load_models():
    try:
        # Load the lung cancer model
        lung_model = joblib.load('models/best_model_lung_cancer.pkl')
        
        # Simulating a Keras model load for demonstration
        brain_model_path = 'models/best_model_brain_tumor.keras'
        # Check if the .keras file exists and load it
        try:
            from tensorflow.keras.models import load_model
            brain_model = load_model(brain_model_path)
        except ImportError:
            st.warning(f"TensorFlow is not installed. Brain tumor model prediction will not be available. Please install TensorFlow to use this feature.")
            brain_model = None

    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}. Make sure the model files are in the same directory as this script.")
        return None, None
    return lung_model, brain_model

lung_model, brain_model = load_models()


st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a model",
    ["Brain Tumor Detection", "Lung Cancer Prediction"])


st.title("Machine Learning Model Application")
st.write("Use the sidebar to select a prediction model.")


if app_mode == "Brain Tumor Detection":
    st.header("Brain Tumor Detection")
    st.write("Upload a brain MRI image (JPG/PNG) to check for a tumor.")

    if brain_model is None:
        st.error("Brain tumor model could not be loaded.")
    else:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption='Uploaded MRI Image.', use_column_width=True)
            st.write("")
            st.write("Classifying...")
            
            # Preprocess the image for the model
            img = np.array(image.resize((150, 150)))
            img = img / 255.0  # Normalize
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            
            # Make prediction
            try:
                prediction = brain_model.predict(img)
                if prediction[0][0] > 0.5:
                    st.error("Prediction: Tumor Detected")
                    st.markdown(f"**Confidence Score:** `{prediction[0][0]:.2f}`")
                else:
                    st.success("Prediction: No Tumor Detected")
                    st.markdown(f"**Confidence Score:** `{1 - prediction[0][0]:.2f}`")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Lung Cancer Prediction Page
elif app_mode == "Lung Cancer Prediction":
    st.header("Lung Cancer Prediction")
    st.write("Enter the patient's data to predict the risk level for lung cancer.")

    if lung_model is None:
        st.error("Lung cancer model could not be loaded.")
    else:
        # Create input fields based on the lung cancer dataset columns
        st.subheader("Patient Information")
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        gender = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")

        st.subheader("Risk Factors (Scale 1-8, with 8 being highest risk)")
        # Mapping for better user experience
        factor_map = {1: 'Very Low', 2: 'Low', 3: 'Slightly Low', 4: 'Moderate', 5: 'Slightly High', 6: 'High', 7: 'Very High', 8: 'Extremely High'}
        
        air_pollution = st.slider("Air Pollution", 1, 8, 5)
        st.info(f"Selected: {air_pollution} - {factor_map[air_pollution]}")
        alcohol_use = st.slider("Alcohol use", 1, 8, 5)
        st.info(f"Selected: {alcohol_use} - {factor_map[alcohol_use]}")
        dust_allergy = st.slider("Dust Allergy", 1, 8, 5)
        st.info(f"Selected: {dust_allergy} - {factor_map[dust_allergy]}")
        occupational_hazards = st.slider("OccuPational Hazards", 1, 8, 5)
        st.info(f"Selected: {occupational_hazards} - {factor_map[occupational_hazards]}")
        genetic_risk = st.slider("Genetic Risk", 1, 8, 5)
        st.info(f"Selected: {genetic_risk} - {factor_map[genetic_risk]}")
        chronic_lung_disease = st.slider("chronic Lung Disease", 1, 8, 5)
        st.info(f"Selected: {chronic_lung_disease} - {factor_map[chronic_lung_disease]}")
        balanced_diet = st.slider("Balanced Diet", 1, 8, 5)
        st.info(f"Selected: {balanced_diet} - {factor_map[balanced_diet]}")
        obesity = st.slider("Obesity", 1, 8, 5)
        st.info(f"Selected: {obesity} - {factor_map[obesity]}")
        smoking = st.slider("Smoking", 1, 8, 5)
        st.info(f"Selected: {smoking} - {factor_map[smoking]}")
        passive_smoker = st.slider("Passive Smoker", 1, 8, 5)
        st.info(f"Selected: {passive_smoker} - {factor_map[passive_smoker]}")
        chest_pain = st.slider("Chest Pain", 1, 8, 5)
        st.info(f"Selected: {chest_pain} - {factor_map[chest_pain]}")
        coughing_of_blood = st.slider("Coughing of Blood", 1, 8, 5)
        st.info(f"Selected: {coughing_of_blood} - {factor_map[coughing_of_blood]}")
        fatigue = st.slider("Fatigue", 1, 8, 5)
        st.info(f"Selected: {fatigue} - {factor_map[fatigue]}")
        weight_loss = st.slider("Weight Loss", 1, 8, 5)
        st.info(f"Selected: {weight_loss} - {factor_map[weight_loss]}")
        shortness_of_breath = st.slider("Shortness of Breath", 1, 8, 5)
        st.info(f"Selected: {shortness_of_breath} - {factor_map[shortness_of_breath]}")
        wheezing = st.slider("Wheezing", 1, 8, 5)
        st.info(f"Selected: {wheezing} - {factor_map[wheezing]}")
        swallowing_difficulty = st.slider("Swallowing Difficulty", 1, 8, 5)
        st.info(f"Selected: {swallowing_difficulty} - {factor_map[swallowing_difficulty]}")
        clubbing_of_finger_nails = st.slider("Clubbing of Finger Nails", 1, 8, 5)
        st.info(f"Selected: {clubbing_of_finger_nails} - {factor_map[clubbing_of_finger_nails]}")
        frequent_cold = st.slider("Frequent Cold", 1, 8, 5)
        st.info(f"Selected: {frequent_cold} - {factor_map[frequent_cold]}")
        dry_cough = st.slider("Dry Cough", 1, 8, 5)
        st.info(f"Selected: {dry_cough} - {factor_map[dry_cough]}")
        snoring = st.slider("Snoring", 1, 8, 5)
        st.info(f"Selected: {snoring} - {factor_map[snoring]}")


        if st.button("Predict Lung Cancer Risk"):
            input_data = [
                age, gender, air_pollution, alcohol_use, dust_allergy,
                occupational_hazards, genetic_risk, chronic_lung_disease,
                balanced_diet, obesity, smoking, passive_smoker,
                chest_pain, coughing_of_blood, fatigue, weight_loss,
                shortness_of_breath, wheezing, swallowing_difficulty,
                clubbing_of_finger_nails, frequent_cold, dry_cough, snoring
            ]
            
            
            input_array = np.array(input_data).reshape(1, -1)
            
            try:
                prediction = lung_model.predict(input_array)
                st.subheader("Prediction Result")
                if prediction[0] == 'High':
                    st.error("The predicted lung cancer risk level is: HIGH")
                elif prediction[0] == 'Medium':
                    st.warning("The predicted lung cancer risk level is: MEDIUM")
                else:
                    st.success("The predicted lung cancer risk level is: LOW")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
