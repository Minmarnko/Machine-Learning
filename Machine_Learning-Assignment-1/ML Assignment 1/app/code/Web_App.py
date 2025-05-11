import streamlit as st
import pickle
import numpy as np

# Load the model and scaler from disk
model_path = './car_selling_price.model'


with open(model_path, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

#input
st.title("Car Price Prediction")

st.write("Enter the car features to predict the selling price:")


year = st.number_input('Year', min_value=1983, max_value=2020, step=1)
km_driven = st.number_input('Kilometer Driven( in KM)', min_value=1000, max_value=2370000, step=1000)
max_power = st.number_input('Max Power (in BHP)', min_value=20.0, max_value=500.0, step=1.0)

if st.button('Predict Selling Price'):
    # Prepare the feature array for prediction
    features = np.array([[year, km_driven, max_power]])
    
    # Predict the selling price
    predicted_price_log = loaded_model.predict(features)
    
    # Reverse the log transformation to get the actual price
    predicted_price = np.exp(predicted_price_log)
    
    # Display the result
    st.write(f"Predicted Selling Price: {predicted_price[0]:,.2f}")
    pass

st.write("This is a prototype app for predicting car selling prices using a pre-trained model.")