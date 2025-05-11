import streamlit as st
import pickle
import os
import numpy as np
import mlflow.pyfunc
from linear_regression import LinearRegression, NormalPenalty, LassoPenalty, RidgePenalty, ElasticPenalty, Normal, Lasso, Ridge, ElasticNet

# Load the old model
filename1 = './models/car_price_old.model'
loaded_data1 = pickle.load(open(filename1, 'rb'))
model_old = loaded_data1['model']
scaler_old = loaded_data1['scaler']
name_map_old = loaded_data1['name_map']
engine_default_old = loaded_data1['engine_default']
mileage_default_old = loaded_data1['mileage_default']

# Load the new model
filename2 = './models/car_price_new.model'
loaded_data2 = pickle.load(open(filename2, 'rb'))
model_new = loaded_data2['model']
scaler_new = loaded_data2['scaler']
name_map_new = loaded_data2['name_map']
engine_default_new = loaded_data2['engine_default']
mileage_default_new = loaded_data2['mileage_default']

# Load classification model and related data
filename_class = './models/Staging/values.pkl'
values_class = pickle.load(open(filename_class, 'rb'))

k_range = values_class['k_range']
scaler_class = values_class['scaler']
poly = values_class['poly']
year_class = values_class['year_default']
max_class = values_class['max_default']

# Load the classification model from MLflow
mlflow.set_tracking_uri("https://mlflow.cs.ait.ac.th/")
model_name = "st124145-a3-model"
model_class = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Staging")

# Prediction function for the old model
def prediction_old(name, engine, mileage):
    sample = np.array([[name, engine, mileage]])
    sample_scaled = scaler_old.transform(sample)
    result = np.exp(model_old.predict(sample_scaled))
    return result

# Prediction function for the new model
def prediction_new(name, engine, mileage):
    sample = np.array([[name, engine, mileage]])
    sample_scaled = scaler_new.transform(sample)
    intercept = np.ones((sample_scaled.shape[0], 1))
    sample_scaled = np.concatenate((intercept, sample_scaled), axis=1)
    result = np.exp(model_new.predict(sample_scaled))
    return result

# Prediction function for the classification model (with numerical features only)
def prediction_class_numerical(max_power, year):
    sample = np.array([[max_power, year]])
    sample = scaler_class.transform(sample)
    sample = np.insert(sample, 0, 1, axis=1)  # Insert an intercept
    sample_poly = poly.transform(sample)      # Polynomial transformation
    result = model_class.predict(sample_poly)
    return k_range[result[0]]

# Streamlit app layout
st.title('Car Price Prediction App')

# Sidebar to choose the model
model_choice = st.sidebar.selectbox('Choose a model', ('Old Model', 'New Model', 'Classification Model'))

st.write(f'You selected the **{model_choice}**.')

if model_choice == 'Old Model':
    st.subheader('Old Model Prediction')
    
    # Input fields for old model
    brand_name = st.text_input('Car Brand', '')
    engine = st.number_input('Engine (in CC)', min_value=500, max_value=5000, step=100, value=int(engine_default_old))
    mileage = st.number_input('Mileage (in KMPL)', min_value=5.0, max_value=50.0, step=0.1, value=float(mileage_default_old))
    
    if st.button('Predict Price'):
        name = name_map_old.get(brand_name, '32')
        result = prediction_old(name, engine, mileage)
        st.write(f'Estimated price: {int(result[0])}')

elif model_choice == 'New Model':
    st.subheader('New Model Prediction')
    
    # Input fields for new model
    brand_name = st.text_input('Car Brand', '')
    engine = st.number_input('Engine (in CC)', min_value=500, max_value=5000, step=100, value=int(engine_default_new))
    mileage = st.number_input('Mileage (in KMPL)', min_value=5.0, max_value=50.0, step=0.1, value=float(mileage_default_new))
    
    if st.button('Predict Price'):
        name = name_map_new.get(brand_name, '32')
        result = prediction_new(name, engine, mileage)
        st.write(f'Estimated price: {int(result[0])}')

else:
    st.subheader('Classification Model Prediction (Numerical Features Only)')
    
    # Input fields for classification model
    year = st.number_input('Year', min_value=1980, max_value=2024, step=1, value=int(year_class))
    max_power = st.number_input('Max Power (in BHP)', min_value=20.0, max_value=500.0, step=1.0, value=float(max_class))
    
    if st.button('Predict Price Range'):
        result = prediction_class_numerical(max_power, year)
        st.write(f'Estimated price range: {result}')
