import streamlit as st
import os
import pickle
import numpy as np
import mlflow.pyfunc
from linear_regression import LinearRegression, NormalPenalty, LassoPenalty, RidgePenalty, ElasticPenalty, Normal, Lasso, Ridge, ElasticNet

# Load the old model
filename1 = 'model/car_price_old.model'
# filename1 = os.path.join('model', 'car_price_old.model')
# print(os.listdir("./model/"))
loaded_data1 = pickle.load(open(filename1, 'rb'))
model_old = loaded_data1['model']
scaler_old = loaded_data1['scaler']
year_default_old = loaded_data1['year_default']
km_driven_default_old = loaded_data1['km_driven_default']
max_power_default_old = loaded_data1['max_power_default']

# Load the new model
filename2 = 'model/car_price_new.model'
loaded_data2 = pickle.load(open(filename2, 'rb'))
model_new = loaded_data2['model']
scaler_new = loaded_data2['scaler']
engine_default_new = loaded_data2['engine_default']
max_power_default_new = loaded_data2['max_power_default']
mileage_default_new = loaded_data2['mileage_default']

# Load classification model and related data
filename_class = 'model/values.pkl'
values_class = pickle.load(open(filename_class, 'rb'))

k_range = values_class['k_range']
scaler_class = values_class['scaler']
poly = values_class['poly']
engine_class = values_class['engine_default']
max_power_class = values_class['max_power_default']
mileage_class = values_class['mileage_default']

os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

# Load the classification model from MLflow
mlflow.set_tracking_uri("http://mlflow.ml.brain.cs.ait.ac.th/")
model_name = "st125437_a3_model"
model_class = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/1")

# Prediction function for the old model
def prediction_old(year, km_driven,max_power):
    sample = np.array([[year, km_driven, max_power]])
    sample_scaled = scaler_old.transform(sample)
    result = np.exp(model_old.predict(sample_scaled))
    return result

# Prediction function for the new model
def prediction_new(engine, max_power,mileage):
    sample = np.array([[engine, max_power, mileage]])
    sample_scaled = scaler_new.transform(sample)
    intercept = np.ones((sample_scaled.shape[0], 1))
    sample_scaled = np.concatenate((intercept, sample_scaled), axis=1)
    result = np.exp(model_new.predict(sample_scaled))
    return result

# Prediction function for the classification model (with numerical features only)
def prediction_class_numerical(engine, max_power,mileage):
    sample = np.array([[engine, max_power, mileage]])
    sample = scaler_class.transform(sample)
    sample = np.insert(sample, 0, 1, axis=1)  # Insert an intercept
    sample_poly = poly.transform(sample)      # Polynomial transformation
    result = model_class.predict(sample_poly)
    return k_range[result[0]]

# Streamlit app
st.title('Car Price Prediction App')

# Sidebar to choose the model
model_choice = st.sidebar.selectbox('Choose a model', ('Old Model', 'New Model', 'Classification Model'))

st.write(f'You selected the **{model_choice}**.')

# Set defaults if inputs are empty
if model_choice == 'Old Model':
    year = st.number_input('Year', min_value=1980, max_value=2024, step=1, value=int(year_default_old))
    km_driven = st.number_input('Kilometer Driven (in KM)', min_value=1000, max_value=2370000, step=1000, value=int(km_driven_default_old))
    max_power = st.number_input('Max Power (in BHP)', min_value=20.0, max_value=500.0, step=1.0, value=float(max_power_default_old))
    if st.button('Predict'):
        result = prediction_old(year, km_driven, max_power)
        st.write(f'Estimated price: {int(result[0])}')
elif model_choice == 'New Model':
    engine = st.number_input('Engine', min_value=50, max_value=5000, step=10, value=int(engine_default_new))
    max_power = st.number_input('Max Power (in BHP)', min_value=20.0, max_value=500.0, step=1.0, value=float(max_power_default_new))
    mileage = st.number_input('Mileage (in KMPL)', min_value=5.0, max_value=50.0, step=0.1, value=float(mileage_default_new))
    if st.button('Predict'):
        result = prediction_new(engine, max_power, mileage)
        st.write(f'Estimated price: {int(result[0])}')
else:
    st.subheader('Classification Model Prediction')
    
    # Input fields for classification model
    engine = st.number_input('Engine', min_value=50, max_value=5000, step=10, value=int(engine_default_new))
    max_power = st.number_input('Max Power (in BHP)', min_value=20.0, max_value=500.0, step=1.0, value=float(max_power_default_new))
    mileage = st.number_input('Mileage (in KMPL)', min_value=5.0, max_value=50.0, step=0.1, value=float(mileage_default_new))
    
    if st.button('Predict Price Range'):
        result = prediction_class_numerical(engine,max_power,mileage)
        st.write(f'Estimated price range: {result}')


