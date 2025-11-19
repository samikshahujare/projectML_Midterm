# Water Potability Prediction API

This API predicts whether water is safe (potable) or unsafe.

## Run Locally (no Docker)
pip install -r requirements.txt
uvicorn app:app --reload

nginx
Copy code

Visit the docs:  
http://127.0.0.1:8000/docs

## Run with Docker
docker build -t water-api .
docker run -p 8000:8000 water-api



Use POST /predict with JSON body.