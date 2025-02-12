import streamlit as st
import pandas as pd
import joblib


model = joblib.load(r"heart_prediction_94%_pickle")  
encoder_dict = joblib.load(r"C:\Users\Admin\Desktop\AI\Heart Disease Machine\encoder_dict.pkl")  


numeric_ranges = {
    "PhysicalHealthDays": (0, 30), 
    "MentalHealthDays": (0, 30),  
    "SleepHours": (0, 24),  
    "HeightInMeters": (0.91, 2.41),  
    "WeightInKilograms": (28.12, 292.57),  
    "BMI": (12.02, 97.65) 
}

questions = {
    "Sex": ["Male", "Female"],
    "GeneralHealth": ["Excellent", "Very good", "Good", "Fair", "Poor"],
    "PhysicalHealthDays": "numeric",
    "MentalHealthDays": "numeric",
    "LastCheckupTime": ["Within past year", "Within past 2 years", "Within past 5 years", "Never"],
    "PhysicalActivities": ["Yes", "No"],
    "SleepHours": "numeric",
    "RemovedTeeth": ["Yes", "No"],
    "HadAngina": ["Yes", "No"],
    "HadStroke": ["Yes", "No"],
    "HadAsthma": ["Yes", "No"],
    "HadSkinCancer": ["Yes", "No"],
    "HadCOPD": ["Yes", "No"],
    "HadDepressiveDisorder": ["Yes", "No"],
    "HadKidneyDisease": ["Yes", "No"],
    "HadArthritis": ["Yes", "No"],
    "HadDiabetes": ["Yes", "No"],
    "DeafOrHardOfHearing": ["Yes", "No"],
    "BlindOrVisionDifficulty": ["Yes", "No"],
    "DifficultyConcentrating": ["Yes", "No"],
    "DifficultyWalking": ["Yes", "No"],
    "DifficultyDressingBathing": ["Yes", "No"],
    "DifficultyErrands": ["Yes", "No"],
    "SmokerStatus": ["Never smoked", "Former smoker", "Current smoker"],
    "ECigaretteUsage": ["Yes", "No"],
    "ChestScan": ["Yes", "No"],
    "RaceEthnicityCategory": ["White", "Black", "Asian", "Hispanic", "Other"],
    "AgeCategory": ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"],
    "HeightInMeters": "numeric",
    "WeightInKilograms": "numeric",
    "BMI": "numeric",
    "AlcoholDrinkers": ["Yes", "No"],
    "HIVTesting": ["Yes", "No"],
    "FluVaxLast12": ["Yes", "No"],
    "PneumoVaxEver": ["Yes", "No"],
    "TetanusLast10Tdap": ["Yes", "No"],
    "HighRiskLastYear": ["Yes", "No"],
    "CovidPos": ["Yes", "No"]
}

model_features = model.feature_names_in_

if "responses" not in st.session_state:
    st.session_state.responses = []
if "question_index" not in st.session_state:
    st.session_state.question_index = 0

st.title("🩺 Heart Disease Prediction")

question_keys = list(questions.keys())
if st.session_state.question_index < len(question_keys):
    question = question_keys[st.session_state.question_index]
    options = questions[question]

    if options == "numeric":
        min_val, max_val = numeric_ranges.get(question, (0, 100))  

        
        if isinstance(min_val, int) and isinstance(max_val, int):
            user_response = st.number_input(f"{question}:", min_value=min_val, max_value=max_val, step=1, value=min_val)
        else:
            user_response = st.number_input(f"{question}:", min_value=float(min_val), max_value=float(max_val), step=0.1, value=float(min_val))

    else:
        user_response = st.radio(f"{question}:", options, index=0)

    if st.button("Next"):
        st.session_state.responses.append(user_response)
        st.session_state.question_index += 1
        st.rerun()

elif len(st.session_state.responses) == len(question_keys):

    user_data = pd.DataFrame([dict(zip(question_keys, st.session_state.responses))])

    for col in user_data.columns:
        if col in encoder_dict:
            user_data[col] = user_data[col].map(lambda x: encoder_dict[col].transform([x])[0] if x in encoder_dict[col].classes_ else -1)

    for feature in model_features:
        if feature not in user_data.columns:
            user_data[feature] = user_data.mode().iloc[0] 

    user_data = user_data[model_features]

    st.write("📊 Data to be passed to the model:", user_data)

    try:
     
        prediction = model.predict(user_data)

       
        if prediction[0] == 1:
            st.error("⚠️ Prediction: High risk of heart disease.")
        else:
            st.success("✅ Prediction: Low risk of heart disease.")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

    if st.button("🔄 Restart"):
        st.session_state.responses = []
        st.session_state.question_index = 0
        st.rerun()

