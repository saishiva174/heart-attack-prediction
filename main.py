import base64
import io
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Force matplotlib to run without a GUI (essential for headless cloud platforms)
matplotlib.use('Agg')

import joblib
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from lime import lime_tabular
from pydantic import BaseModel

# --- 🤫 SILENCE VOLATILE MACHINE LEARNING WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="An ill-conditioned matrix detected")


# --- 🔑 ENVIRONMENT & CORE CONFIGURATION ---
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


# --- 🛡️ CORS CONFIGURATION FOR LOCAL SITES & PRODUCTION LOOPBACKS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://127.0.0.2:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- ⚙️ PERSISTED ML MODELS & COLUMN STRUCTURES ---
model = joblib.load("heart_disease_voting_ensemble.pkl")

ordered_columns = [
    "Age", "Sex", "chest pain type", "resting bp s", "cholesterol",
    "fasting blood sugar", "resting ecg", "max heart rate", 
    "exercise angina", "oldpeak", "ST slope"
]


# --- 📊 LIME ENVIRONMENT STATISTICS CONFIGURATION ---
background_data_path = "lime_training_background.csv"

if os.path.exists(background_data_path):
    bg_df = pd.read_csv(background_data_path)
    training_background = bg_df.values  
else:
    training_background = np.zeros((1, 11))

lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=training_background,
    feature_names=ordered_columns,
    class_names=['Normal', 'Heart Disease'],
    mode='classification',
    categorical_features=[1, 2, 5, 6, 8, 10],
    verbose=False
)


# --- 🤖 LANGCHAIN MISTRAL AI INITIALIZATION ---
# Using LangChain's dedicated ChatMistralAI interface class
ai_client = ChatMistralAI(
    model="mistral-small-latest", 
    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0.2,
    max_tokens=600
)


# --- 📋 VALIDATION MODEL DATA SCHEMAS ---
class PatientData(BaseModel):
    Age: float
    Sex: int
    chest_pain_type: int
    resting_bp_s: float
    cholesterol: float
    fasting_blood_sugar: float  
    resting_ecg: int
    max_heart_rate: float
    exercise_angina: int
    oldpeak: float
    ST_slope: int


class ChatMessage(BaseModel):
    role: str
    text: str


class ChatQuery(BaseModel):
    message: str
    history: list[ChatMessage]
    age: float
    sex: int
    chest_pain_type: int
    resting_bp_s: float
    cholesterol: float
    fasting_blood_sugar: float
    resting_ecg: int
    max_heart_rate: float
    exercise_angina: int
    oldpeak: float
    ST_slope: int
    risk_percentage: float
    shap_summary: str
    lime_summary: str


# --- 📡 ENDPOINT ROUTING ---

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "Heart Diagnostics Pipeline Active"}


@app.post("/predict")
def predict_risk(data: PatientData):
    fbs_flag = 1 if data.fasting_blood_sugar > 120 else 0
    adjusted_slope = data.ST_slope - 1

    input_df = pd.DataFrame([[
        data.Age, data.Sex, data.chest_pain_type, data.resting_bp_s, data.cholesterol,
        fbs_flag, data.resting_ecg, data.max_heart_rate, data.exercise_angina, 
        data.oldpeak, adjusted_slope
    ]], columns=ordered_columns)

    proba = model.predict_proba(input_df)[0][1]


    plt.close('all')
    plt.clf()
    # A. GENERATE SHAP ARRAYS
    xgb_estimator = model.named_estimators_['xgb']
    shap_explainer = shap.TreeExplainer(xgb_estimator)
    shap_values = shap_explainer(input_df)
    
    vals = shap_values.values[0]
    shap_dict = {col: round(float(val), 4) for col, val in zip(ordered_columns, vals)}
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    shap_summary_text = ", ".join([f"{col}: {val}" for col, val in sorted_shap])

    fig_shap, ax_shap = plt.subplots(figsize=(8, 3.5))
    shap.plots.waterfall(shap_values[0], show=False)
    
    buf_shap = io.BytesIO()
    plt.savefig(buf_shap, format="png", bbox_inches="tight")
    plt.close(fig_shap)
    buf_shap.seek(0)
    shap_base64 = base64.b64encode(buf_shap.read()).decode("utf-8")

    # B. GENERATE LIME RULE DEVIATIONS
    exp = lime_explainer.explain_instance(
        data_row=input_df.iloc[0].values,
        predict_fn=model.predict_proba,
        num_features=5
    )
    lime_list = exp.as_list()[:3]
    lime_summary_text = ", ".join([f"Rule ({rule}): impact {round(weight, 4)}" for rule, weight in lime_list])
    
    plt.clf() 
    fig_lime = exp.as_pyplot_figure()
    
    buf_lime = io.BytesIO()
    plt.savefig(buf_lime, format="png", bbox_inches="tight")
    plt.close(fig_lime)
    buf_lime.seek(0)
    lime_base64 = base64.b64encode(buf_lime.read()).decode("utf-8")

    return {
        "risk_percentage": round(proba * 100, 2),
        "shap_plot": f"data:image/png;base64,{shap_base64}",
        "lime_plot": f"data:image/png;base64,{lime_base64}",
        "shap_summary": shap_summary_text,
        "lime_summary": lime_summary_text
    }


@app.post("/chat")
def chat_bot(query: ChatQuery):
    sex_label = "Male" if query.sex == 1 else "Female"
    cp_label = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal Pain", 4: "Asymptomatic"}.get(query.chest_pain_type, "Unknown")
    ecg_label = {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"}.get(query.resting_ecg, "Unknown")
    slope_label = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}.get(query.ST_slope, "Unknown")
    angina_label = "Yes" if query.exercise_angina == 1 else "No"
    fbs_flag = "High (>120 mg/dl)" if query.fasting_blood_sugar > 120 else "Normal (<=120 mg/dl)"

    # High-density instruction block packed inside a LangChain SystemMessage wrapper
    # High-performance, token-optimized RAG system prompt framework
    rag_context = f"""
    [ROLE]
    You are an expert, compassionate Medical AI Explainer. Your task is to interpret a soft-voting ensemble Machine Learning model's heart attack risk prediction for a patient.

    [PATIENT CLINICAL PROFILE]
    - Age: {query.age} years old
    - Biological Sex: {sex_label}
    - Chest Pain Type: {cp_label}
    - Resting Blood Pressure: {query.resting_bp_s} mmHg
    - Serum Cholesterol: {query.cholesterol} mg/dl
    - Fasting Blood Sugar: {fbs_flag}
    - Resting ECG Results: {ecg_label}
    - Max Heart Rate Achieved: {query.max_heart_rate} bpm
    - Exercise-Induced Angina: {angina_label}
    - ST Depression (Oldpeak): {query.oldpeak}
    - Peak Exercise ST Slope: {slope_label}

    [DIAGNOSTIC MODEL METRICS]
    - Calculated Risk Probability: {query.risk_percentage}% chance of Heart Disease.
    - SHAP Global Feature Impact (Top 3): [{query.shap_summary}]
    - LIME Local Decision Rules (Top 3): [{query.lime_summary}]

    [EXECUTION MANDATES & RULES]
    1. TONALITY: Empathetic, clear, professional, and accessible. Avoid dense mathematical jargon.
    2. INTERPRETATION: Translate the SHAP and LIME metrics into plain English. (e.g., instead of saying "SHAP value for cholesterol is +0.12", say "Your cholesterol level of {query.cholesterol} pushed the model's risk calculation upward").
    3. STRUCTURE: Use clean markdown spacing and bullet points. Never output raw '#' or '*' symbols without proper spacing.
    4. LENGTH: Be thorough but concise. Fully complete all thoughts—do not cut off mid-sentence.
    5. CRITICAL MEDICAL DISCLAIMER: Always conclude your response with a prominent, strict disclaimer stating that you are an AI assistant explaining a statistical machine learning tool, not a doctor, and this output must be verified by a cardiologist.
    """

    # 1. Initialize LangChain Message Array with the System context
    messages = [SystemMessage(content=rag_context)]
    
    # 2. Extract recent back-and-forth context limits
    recent_history = query.history[-4:] if len(query.history) > 4 else query.history
    
    # 3. Append historical values as dynamic LangChain message formats
    for msg in recent_history:
        if msg.role == "model":
            messages.append(AIMessage(content=msg.text))
        else:
            messages.append(HumanMessage(content=msg.text))
            
    # 4. Append current user question statement
    messages.append(HumanMessage(content=query.message))

    try:
        # Use LangChain's uniform .invoke interface handler method
        response = ai_client.invoke(messages)
        reply = response.content
        
    except Exception as e:
        error_str = str(e)
        print(f"LangChain Mistral API Error: {error_str}")
        
        if "429" in error_str:
            reply = "⚠️ [Mistral Notice]: Rate limit hit. Please pause briefly before sending your next query."
        else:
            reply = "I encountered an issue processing the conversation context arrays."

    return {"reply": reply}


# --- 🚀 RUN LOCAL SYSTEM LIFECYCLE ---
if __name__ == "__main__":
    import uvicorn
    
    host_ip = os.getenv("HOST", "127.0.0.1")
    port_no = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host=host_ip, port=port_no, reload=True)