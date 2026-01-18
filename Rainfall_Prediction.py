import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Rainfall Prediction", page_icon=":bar_chart:", layout="wide")
st.title("Rainfall Prediction System")
st.write("Rainfall Prediction Project")
states = ["Maharastra","Kerala","Tamil Nadu","Rajastan","Punjab"]
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct"]
data = []



for state in  states:
    for month_index,month in enumerate(months):
        for _ in range(4):
            if(month in ["Jun","Jul","Aug","Sep"]):
                rainfall = 150 + month_index*5
                humidity = 75
                temp = 25
            else:
                rainfall = 20 + month_index*2
                humidity = 45
                temp = 35

            if state == "Kerala":
                rainfall += 40
            elif state =="Rajastan":
                rainfall -= 15
            elif state =="Tamil Nadu":
                rainfall += 30

            wind_speed = 10 + month_index

            data.append(
                [state,month,humidity,temp,wind_speed,rainfall]
            )

pred_data = pd.DataFrame(data,columns=["State","Month","Avg_Humidity","Avg_Temperature","Avg_Wind Speed","Rainfall_output"])

if(st.checkbox("Show Data")):
    st.write(pred_data.head(10))
    st.write("Total Records: "+str(pred_data.shape[0]))
    st.write("Total Features: "+str(pred_data.shape[1]))
    st.write(pred_data.describe())
# .get_dummies() Converts categorical variables into numerical (binary) columns using one-hot encoding.
# columns=["State","Month"] Specifies that only the State and Month columns should be encoded.
# Each unique value in State or Month becomes a new column like Month_Jan, State_Kerala
# ---------> data_encoded = pd.get_dummies(pred_data,columns=["State","Month"])
# drop_first=True is used to avoid a serious statistical issue called dummy variable trap
# and to make models mathematically stable and efficient. 
# It removes one dummy column per categorical feature.
data_encoded = pd.get_dummies(pred_data,drop_first=True)

x = data_encoded.drop("Rainfall_output",axis=1)
y = data_encoded["Rainfall_output"]


# train_test_split is a utility function used to split datasets into training(seen data) and testing sets(unseen data).
# test_size=0.2
#   20% of the dataset is reserved for testing
#   80% is used for training


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=False)

logistic_model = LinearRegression()
pred_y = logistic_model.fit(x_train,y_train)


st.subheader("Enter Climatic details: ")

state_input = st.selectbox("Select state", states)
month_input = st.selectbox("Select month", months)

temp_input = st.slider("Average temperature (C)", 20,45,30)
humidity_input = st.slider("Average humidity (%)", 30,90,60)
wind_input = st.slider("Average wind (Km/h)", 5,25,12)


input_data = pd.DataFrame([{
    "State": state_input,
    "Month": month_input,
    "Avg_Temperature": temp_input,
    "Avg_Humidity": humidity_input,
    "Avg_Wind Speed": wind_input
}])

input_encoded = pd.get_dummies(input_data,drop_first=True)
input_encoded = input_encoded.reindex(columns=x.columns,fill_value=0)


# [0] — Why it is needed --> .predict() always returns an array 
# [0] extracts the first (and only) value 
# Without [0], prediction would be: array([123.456789])
# With [0], prediction would be: 123.456789

if st.button("Predict Rainfall"):
    prediction = logistic_model.predict(input_encoded)[0]
    st.success(f"Predicted Rainfall: {prediction:.2f} mm")

# :.2f formats the number to 2 decimal places Eg: 123.456789 → 123.46
# st.success(...) Displays a green success message in Streamlit to show final results or outputs

st.write("-------")
st.subheader("Model Evaluation")
st.caption("Antony Thomas | Synthetic India Climate Data | Linear Regression")