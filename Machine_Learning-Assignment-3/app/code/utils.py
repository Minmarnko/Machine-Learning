import os
import pickle
import mlflow
import numpy as np

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

def calculate_y_hardcode(x_1, x_2, submit):
    print(x_1)
    print(x_2)
    print(submit)
    return x_1 + x_2

# Testing function for the classification model
def prediction_class_test(engine, max_power,mileage):
    sample = np.array([[engine, max_power, mileage]])
    sample = scaler_class.transform(sample)
    sample = np.insert(sample, 0, 1, axis=1)  # Insert an intercept
    sample_poly = poly.transform(sample)      # Polynomial transformation
    result = model_class.predict(sample_poly)
    return sample_poly, result