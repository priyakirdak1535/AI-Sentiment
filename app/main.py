from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# Get base directory (important for correct file path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model and vectorizer
model = joblib.load(os.path.join(BASE_DIR, "model/model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "model/vectorizer.pkl"))

# Create FastAPI app
app = FastAPI()

# Input format
class TextInput(BaseModel):
    text: str

# Home route
@app.get("/")
def home():
    return {"message": "Sentiment API is running 🚀"}

# Health check route
@app.get("/health")
def health():
    return {"status": "OK"}

# Prediction route
@app.post("/predict")
def predict(data: TextInput):
    text = [data.text]
    vector = vectorizer.transform(text)
    prediction = model.predict(vector)[0]

    # Convert output to readable format
    if prediction == 1:
        result = "Positive 😊"
    else:
        result = "Negative 😡"

    return {"sentiment": result}