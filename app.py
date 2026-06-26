from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# set streamlit layout wide
st.set_page_config(layout = "wide")
#load the trained models
 with open("best_model.pkl","rb") as file:
   model = pickle.load(file)
# load the MinMaxScaler
with open("scaler.pkl","rb") as file:
  model = pickle.load(file)

# define the input features for the model
feature_names = [
  "CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary","Geography_France","Geography_Germany",
  "Geography_Spain","Gender_Female","Gender_Male","HasCrCard_0","HasCrCard_1","IsActiveMember_0","IsActiveMember_1"
]

# Columns requiring scaling
scale_vars = ["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]

# Updated default values
default_values = [600,30,2,8000,2,60000,
                  True,False,False,True,False,False,True,False,True
                 ]

# sidebar settu
st.sidebar.image("https://daxg39y63pxwu.cloudfront.net/images/blog/churn-models/Customer_Churn_Prediction_Models_in_Machine_Learning.png",use_column_widh = True)
st..sidebar.header("User Inputs")

# colect User inputs
user_inputs = {}
for i,feature in enumerate(feature_names):
  if feature in scale_vars:
    user_input[feature] = st.sidebar.number_input(feature,value = default_values[i]
                                                  step = 1 if isinstance(default_values[i],int) else 0.01
    )
  elif isinstance(default_values[i],bool):
    user_input[feature] = st.sidebar.checkbox(feature,value = default_values[i])

# convert inputs to dataframe
input_data = pd.DataFrame([user_inputs])

# apply MinMaxScaler to the required columns

input_data_scaled = input_data.copy()
input_data_scaled[scale_vars] = scaler.transform(input_data[scale_vars])

# App header

st.image("https://www.alamy.com/customer-management-image335850425.html",use_column_widh = True)
st.title("👨🏻‍💻 Customer Churn Prediction")

# page layout
left_col,right_col = st.columns(2)

# left page Feature importance
st.header("Feature Importance")
# Load feature importance data from the excel file
feature_importance_df = pd.read_excel("feature_importance.xlsx",usecols = ["feature","Feature Importance Score"])
#plot the feature importance chat
fig = px.bar(
  feature_importance_df.sort_values(by = "Feature Importance Score",ascending = False),
  x="Feature Importance Score",
  y="Feature",
  orientation = "h",
  title = "Feature Importance",
  labels = {"Feature Importance score": "Importance","Feature":"Features"},
  widh = 400,
  height = 500
)
st.plotly_chart(fig)

# Right page: prediction
with right_col:
  st.header("prediction")
  if st.button("predict"):
    #get the predicted probabilities and label
    probabilities = model.predict_proba(input_data_scaled)[0]
    prediction = model.predict(input_data_scaled)[0]
    # map prediction to label
    prediction_label = "Churned" if prediction ==1 else "Retain"

#display results
    st.subheader(f"Predicted Value: {prediction_label}")
    st.write(f"Predicted Probability: {probabilities[1]:.2%} (Churn)")
    st.write(f"Predicted Probability: {probabilities[0]:.2%} (Retain)")
    #display the clear output for the prediction
    st.markdown(f"## Output: **{prediction_label}**")
