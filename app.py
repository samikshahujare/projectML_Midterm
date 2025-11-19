from fastapi import FastAPI
from pydantic import BaseModel, Field
import xgboost as xgb
import numpy as np

# Load trained model (water.json)
model = xgb.XGBClassifier()
model.load_model("water.json")

app = FastAPI(
    title="Water Potability Classifier API",
    description="Predicts whether water is potable or not.",
    version="1.0"
)

@app.get("/")
def home():
    return {"message": "Water Potability API is running. Visit /docs for API testing."}

class WaterInput(BaseModel):
    ph: float = Field(..., example=7.2)
    Hardness: float = Field(..., example=214.0)
    Solids: float = Field(..., example=15000.0)
    Chloramines: float = Field(..., example=7.5)
    Sulfate: float = Field(..., example=300.0)
    Conductivity: float = Field(..., example=400.0)
    Organic_carbon: float = Field(..., example=10.0)
    Trihalomethanes: float = Field(..., example=80.0)
    Turbidity: float = Field(..., example=3.5)

@app.post("/predict")
def predict_water(data: WaterInput):
    arr = np.array([[  
        data.ph,
        data.Hardness,
        data.Solids,
        data.Chloramines,
        data.Sulfate,
        data.Conductivity,
        data.Organic_carbon,
        data.Trihalomethanes,
        data.Turbidity
    ]])

    pred = model.predict(arr)[0]

    return {
        "prediction": "Water is POTABLE (SAFE to drink)"
        if pred == 1
        else "Water is NOT POTABLE (UNSAFE to drink)"
    }
