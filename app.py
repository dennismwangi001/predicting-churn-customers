import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. Load Models and Scaler using Pickle
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as file:
        return pickle.load(file)

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
    scaler = load_scaler()
except FileNotFoundError:
    st.error("Error: 'best_model.pkl' or 'scaler.pkl' not found. Please run the notebook cell to save them.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Define Features and Defaults
# -----------------------------------------------------------------------------
# These must match the columns after pd.get_dummies in the notebook
feature_names = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary",
    "Geography_France", "Geography_Germany", "Geography_Spain",
    "Gender_Female", "Gender_Male",
    "HasCrCard_0", "HasCrCard_1",
    "IsActiveMember_0", "IsActiveMember_1"
]

# Columns that were scaled in the notebook
scale_vars = ["CreditScore","Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]

# Default values for inputs
default_values = {
    "CreditScore": 600,
    "Age": 30,
    "Tenure": 2,
    "Balance": 8000.0,
    "NumOfProducts": 2,
    "EstimatedSalary": 60000.0,
    "Geography_France": True,
    "Geography_Germany": False,
    "Geography_Spain": False,
    "Gender_Female": True,
    "Gender_Male": False,
    "HasCrCard_0": False,
    "HasCrCard_1": True,
    "IsActiveMember_0": False,
    "IsActiveMember_1": True
}

# -----------------------------------------------------------------------------
# 4. Sidebar Inputs
# -----------------------------------------------------------------------------
st.sidebar.image(
    "https://daxg39y63pxwu.cloudfront.net/images/blog/churn-models/Customer_Churn_Prediction_Models_in_Machine_Learning.png",
    width=True
)
st.sidebar.header("User Inputs")

user_inputs = {}

for feature in feature_names:
    default_val = default_values[feature]
    
    if feature in scale_vars:
        # Numeric Input for scalable features
        user_inputs[feature] = st.sidebar.number_input(
            feature,
            value=float(default_val) if isinstance(default_val, int) else default_val,
            step=1.0 if isinstance(default_val, int) else 100.0
        )
    elif isinstance(default_val, bool):
        # Checkbox for boolean/categorical features
        user_inputs[feature] = st.sidebar.checkbox(feature, value=default_val)
    else:
        # Fallback for other numeric types
        user_inputs[feature] = st.sidebar.number_input(feature, value=default_val)

# -----------------------------------------------------------------------------
# 5. Data Processing & Scaling
# -----------------------------------------------------------------------------
# Create DataFrame from inputs
input_df = pd.DataFrame([user_inputs], columns=feature_names)

# Prepare data for scaling
# We extract ONLY the columns needed for scaling and convert to NumPy array.
# This bypasses Sklearn's strict feature name validation during transform if versions differ,
# but ensures the values are scaled correctly based on the training data min/max.
try:
    raw_scale_data = input_df[scale_vars].values  # .values returns a numpy array
    
    # Transform the numeric data using the fitted scaler
    scaled_data = scaler.transform(raw_scale_data)
    
    # Create a new DataFrame with scaled values, keeping original non-scaled columns
    input_scaled_df = input_df.copy()
    input_scaled_df[scale_vars] = scaled_data
    
except Exception as e:
    st.error(f"Error during data scaling: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 6. Main UI Layout
# -----------------------------------------------------------------------------
st.image("https://dqy38fnwh4fqs.cloudfront.net/UHA9NGE97B68ANGFA7OLGRK8K6JB/projects/telecom-customer-churn-analysis11d3d0af-4144-48e4-8a03-0af760054dbd.webp", width=True)
st.title("👨🏻‍💻 Customer Churn Prediction App")

left_col, right_col = st.columns(2)

# --- Left Column: Feature Importance ---
with left_col:
    st.header("Feature Importance")
    try:
        # Load feature importance data
        feat_imp_df = pd.read_excel("feature_importance.xlsx")
        
        # Ensure we have the right columns
        if "feature" in feat_imp_df.columns and "Feature Importance Score" in feat_imp_df.columns:
            fig = px.bar(
                feat_imp_df.sort_values(by="Feature Importance Score", ascending=False),
                x="Feature Importance Score",
                y="feature",
                orientation="h",
                title="Top Factors Influencing Churn",
                color="Feature Importance Score",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Excel file columns do not match expected format.")
            
    except FileNotFoundError:
        st.warning("File 'feature_importance.xlsx' not found. Skipping chart.")
    except Exception as e:
        st.error(f"Could not load feature importance: {e}")

# --- Right Column: Prediction ---
with right_col:
    st.header("Prediction Result")
    
    # Display current input summary
    with st.expander("View Input Data"):
        st.dataframe(input_df.T, hide_index=True)

    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        try:
            # Perform Prediction
            # The model expects the DataFrame with the same column order as X_train
            prediction = model.predict(input_scaled_df)[0]
            probabilities = model.predict_proba(input_scaled_df)[0]
            
            # Determine Label
            label = "Churned 😟" if prediction == 1 else "Retained 😊"
            churn_prob = probabilities[1]
            retain_prob = probabilities[0]
            
            # Display Results
            st.markdown(f"### Prediction: {label}")
            
            # Progress bars for probability
            st.progress(float(churn_prob), text=f"Churn Probability: {churn_prob:.2%}")
            st.progress(float(retain_prob), text=f"Retention Probability: {retain_prob:.2%}")
            
            # Color-coded result box
            if prediction == 1:
                st.error("**High Risk:** This customer is likely to leave.")
            else:
                st.success("**Low Risk:** This customer is likely to stay.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)