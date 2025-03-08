from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

# Load the trained model
with open("failure_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize FastAPI app
app = FastAPI()

# Define the input data format
class FailureInput(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    # Add more features as per your dataset

@app.get("/")
def home():
    return {"message": "Hardware Failure Prediction API is Running!"}

@app.post("/predict")
def predict_failure(data: FailureInput):
    input_data = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    result = "Failure Expected" if prediction == 1 else "No Failure Expected"
    return {"prediction": result}
