import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go  # Required for gauge charts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
from openai import OpenAI
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="AI Churn Predictor & Retention Advisor", layout="wide")
st.title("🔮 Customer Churn Prediction & AI Retention Advisor")

# --- Sidebar Inputs & Controls ---
st.sidebar.header("️ Model & Business Settings")

# API Key Input for AI Recommendations
api_key = st.sidebar.text_input("OpenAI API Key", type="password", 
                                help="Required for AI-generated retention strategies.")

# Business Risk Thresholds
st.sidebar.subheader("🎯 Business Risk Thresholds")
high_risk_threshold = st.sidebar.slider("High Risk Threshold (%)", 50, 90, 70)
medium_risk_threshold = st.sidebar.slider("Medium Risk Threshold (%)", 20, 49, 30)

# Model Hyperparameters
st.sidebar.subheader("🤖 Model Hyperparameters")
n_estimators = st.sidebar.slider("Number of Trees (RF)", 50, 500, 200, step=50)
max_depth = st.sidebar.slider("Max Depth", 2, 20, 10)
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, step=0.05)

# Data Exploration Toggles
st.sidebar.subheader("📊 Data View Options")
show_raw_data = st.sidebar.checkbox("Show Raw Dataset")
show_feature_dist = st.sidebar.checkbox("Show Feature Distributions")
show_feature_importance = st.sidebar.checkbox("Show Feature Importance")

# Export Options
st.sidebar.subheader("💾 Export Results")
if st.sidebar.button("Export Predictions to CSV"):
    if 'predictions_df' in st.session_state:
        csv = st.session_state['predictions_df'].to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.sidebar.warning("No predictions to export yet.")

# --- Data Loading & Preprocessing (Cached) ---
@st.cache_data
def load_data():
    if not os.path.exists("Churn_Modelling.csv"):
        st.error("❌ 'Churn_Modelling.csv' not found in the current directory.")
        return None
    return pd.read_csv("Churn_Modelling.csv")

@st.cache_resource
def prepare_and_train_model(n_est, max_d, t_size):
    df = load_data()
    if df is None:
        return None, None, None, None, 0, [], None
    
    # Drop non-predictive columns
    drop_cols = ['RowNumber', 'CustomerId', 'Surname']
    X = df.drop(columns=drop_cols + ['Exited'])
    y = df['Exited']
    
    # Encode Categoricals
    le_geo = LabelEncoder()
    le_gender = LabelEncoder()
    X['Geography'] = le_geo.fit_transform(X['Geography'])
    X['Gender'] = le_gender.fit_transform(X['Gender'])
    
    # Scale Features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=t_size, random_state=42, stratify=y
    )
    
    # Train Model
    model = RandomForestClassifier(
        n_estimators=n_est, 
        max_depth=max_d, 
        random_state=42, 
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, scaler, le_geo, le_gender, acc, X.columns.tolist(), df

# Initialize Model
model, scaler, le_geo, le_gender, accuracy, feature_names, df = prepare_and_train_model(
    n_estimators, max_depth, test_size
)

if model is None:
    st.stop()

st.sidebar.success(f"✅ Model Accuracy: {accuracy:.2%}")

# --- Main Dashboard Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Predict Individual", " Analytics Dashboard", "🤖 AI Retention Advisor", "📈 Feature Insights"])

# TAB 1: PREDICT INDIVIDUAL CUSTOMER
with tab1:
    st.header("Predict Single Customer Churn")
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input("Credit Score", 300, 900, 600)
        age = st.number_input("Age", 18, 92, 35)
        tenure = st.number_input("Tenure (Years)", 0, 10, 5)
        balance = st.number_input("Account Balance", 0.0, 250000.0, 75000.0)
        num_products = st.selectbox("Num Of Products", [1, 2, 3, 4], index=0)
        
    with col2:
        has_cr_card = st.selectbox("Has Credit Card?", ["No", "Yes"], index=1)
        is_active_member = st.selectbox("Is Active Member?", ["No", "Yes"], index=1)
        estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 100000.0)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"], index=0)
        gender = st.selectbox("Gender", ["Female", "Male"], index=0)
        
    if st.button(" Predict Churn Probability", type="primary"):
        # Prepare input vector
        geo_enc = le_geo.transform([geography])[0]
        gen_enc = le_gender.transform([gender])[0]
        cr_card = 1 if has_cr_card == "Yes" else 0
        active_mem = 1 if is_active_member == "Yes" else 0
        
        input_data = pd.DataFrame([[credit_score, geo_enc, gen_enc, age, tenure, 
                                    balance, num_products, cr_card, active_mem, 
                                    estimated_salary]], columns=feature_names)
        
        proba = model.predict_proba(input_data)[0][1] * 100
        
        # Determine risk level based on thresholds
        if proba >= high_risk_threshold:
            risk_level = "HIGH RISK"
            color = "#d32f2f"
        elif proba >= medium_risk_threshold:
            risk_level = "MEDIUM RISK"
            color = "#fbc02d"
        else:
            risk_level = "LOW RISK"
            color = "#388e3c"
        
        st.divider()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Churn Risk", f"{proba:.1f}%")
            st.caption(f"**{risk_level}** (Thresholds: High ≥ {high_risk_threshold}%, Medium ≥ {medium_risk_threshold}%)")
        with c2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba,
                title={'text': "Risk Level"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, medium_risk_threshold], 'color': "#c8e6c9"},
                        {'range': [medium_risk_threshold, high_risk_threshold], 'color': "#fff9c4"},
                        {'range': [high_risk_threshold, 100], 'color': "#ffcdd2"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': high_risk_threshold
                    }
                }
            ))
            
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Store prediction for export
        pred_row = {
            'CreditScore': credit_score, 'Age': age, 'Tenure': tenure,
            'Balance': balance, 'NumOfProducts': num_products,
            'HasCrCard': has_cr_card, 'IsActiveMember': is_active_member,
            'EstimatedSalary': estimated_salary, 'Geography': geography,
            'Gender': gender, 'ChurnProbability': round(proba, 2),
            'RiskLevel': risk_level
        }
        
        if 'predictions_df' not in st.session_state:
            st.session_state['predictions_df'] = pd.DataFrame([pred_row])
        else:
            st.session_state['predictions_df'] = pd.concat([
                st.session_state['predictions_df'], 
                pd.DataFrame([pred_row])
            ], ignore_index=True)

# TAB 2: ANALYTICS DASHBOARD
with tab2:
    st.header("Dataset Insights")
    
    if show_raw_data:
        st.dataframe(df.head(50), use_container_width=True)
        
    if show_feature_dist:
        c1, c2 = st.columns(2)
        with c1:
            fig_age = px.histogram(df, x="Age", color="Exited", barmode="overlay", 
                                   title="Age Distribution by Churn")
            st.plotly_chart(fig_age, use_container_width=True)
        with c2:
            fig_bal = px.histogram(df, x="Balance", color="Exited", barmode="overlay",
                                   title="Balance Distribution by Churn")
            st.plotly_chart(fig_bal, use_container_width=True)
            
    # Overall Stats
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Customers", len(df))
    m2.metric("Churn Rate", f"{df['Exited'].mean():.2%}")
    m3.metric("Avg Tenure", f"{df['Tenure'].mean():.1f} yrs")

# TAB 3: AI RETENTION ADVISOR
with tab3:
    st.header("🤖 AI-Powered Retention Strategy Generator")
    st.caption("Enter your OpenAI API key in the sidebar to activate personalized recommendations.")
    
    ai_prompt_text = st.text_area(
        "Custom Context (Optional)", 
        placeholder="e.g., This customer recently complained about high fees...",
        height=80
    )
    
    # FIXED SYNTAX ERROR: The elif is now correctly nested inside the button's if block
    if st.button("✨ Generate AI Recommendations", disabled=not api_key):
        try:
            client = OpenAI(api_key=api_key)
            
            # Get feature importance for context
            importances = model.feature_importances_
            top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:3]
            feat_context = ", ".join([f"{f} ({i:.2f})" for f, i in top_features])
            
            system_prompt = """You are an expert Customer Success Strategist specializing in banking/telecom churn reduction. 
            Analyze the provided customer profile and churn risk score. Provide 3 specific, actionable retention strategies.
            Format output as: 
            🔴 RISK ASSESSMENT: [Brief summary]
            💡 STRATEGY 1: [Actionable tactic]
            💡 STRATEGY 2: [Actionable tactic] 
            💡 STRATEGY 3: [Actionable tactic]
            Keep it concise, professional, and directly tied to the data points."""
            
            user_prompt = f"""
            Customer Profile: Age={age}, Geography={geography}, Gender={gender}, 
            CreditScore={credit_score}, Tenure={tenure}, Balance={balance}, 
            Products={num_products}, IsActive={is_active_member}, Salary={estimated_salary}
            
            Predicted Churn Risk: {proba:.1f}%
            Top Influencing Features: {feat_context}
            Additional Context: {ai_prompt_text if ai_prompt_text else 'None'}
            """
            
            with st.spinner("AI is analyzing customer profile..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3
                )
                
            st.markdown(response.choices[0].message.content)
            
        except Exception as e:
            st.error(f"❌ AI Error: {str(e)}")
            st.info("Check your API key and ensure you have access to GPT-4o-mini or gpt-3.5-turbo.")

    elif not api_key:
        st.warning("⚠️ Please add your OpenAI API Key in the sidebar to use the AI Advisor.")

# TAB 4: FEATURE INSIGHTS
with tab4:
    st.header("📈 Model Feature Importance")
    
    if show_feature_importance:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig_imp = px.bar(
            x=[feature_names[i] for i in indices],
            y=[importances[i] for i in indices],
            labels={'x': 'Feature', 'y': 'Importance'},
            title="Top Features Driving Churn Predictions"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.caption("Features are ranked by their contribution to the Random Forest model's decision-making process.")