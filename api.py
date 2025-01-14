import os
import tensorflow as tf
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from fastapi.responses import FileResponse

# Initialize FastAPI
api = FastAPI()

# CORS Middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (use specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Handle the root path ("/")
@api.get("/")
def read_root():
    return {"message": "Welcome to the Dress Recommendation API!"}

# Handle favicon.ico request
@api.get("/favicon.ico")
def get_favicon():
    # Serve the favicon.ico directly from the same directory where the api.py file is located
    return FileResponse("favicon.ico")

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
ASOS_API_KEY = os.getenv("ASOS_API_KEY")

# Ensure keys are available
if not OPENAI_API_KEY or not PEXELS_API_KEY or not ASOS_API_KEY:
    raise ValueError("One or more API keys are missing. Check your environment variables.")

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="DRM_quantized.tflite")
interpreter.allocate_tensors()

# Define request and response schemas
class PredictionInput(BaseModel):
    image_url: str
    occasion: str

class PredictionOutput(BaseModel):
    category: str
    confidence: float
    recommendations: list
    trend_images: list
    openai_description: str

# Helper functions
def fetch_trending_images():
    headers = {"Authorization": PEXELS_API_KEY}
    response = requests.get("https://api.pexels.com/v1/curated", headers=headers, params={"per_page": 5})
    if response.status_code == 200:
        return [img["src"]["medium"] for img in response.json().get("photos", [])]
    return []

def fetch_asos_recommendations(category, occasion):
    headers = {"Authorization": ASOS_API_KEY}
    response = requests.get(
        f"https://api.asos.com/recommendations/{category}?occasion={occasion}", headers=headers
    )
    if response.status_code == 200:
        items = response.json().get("items", [])
        return [{"name": item["name"], "image": item["imageUrl"]} for item in items]
    return []

def generate_openai_description(category, occasion):
    openai.api_key = OPENAI_API_KEY
    prompt = f"Provide a detailed fashion-forward description for {category} outfits suitable for a {occasion}."
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error generating description: {e}"

# Prediction API endpoint
@api.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    # Fetch the image
    response = requests.get(input_data.image_url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch image")
    image_data = response.content

    # Preprocess the image
    image = tf.image.decode_image(image_data, channels=3)
    image = tf.image.resize(image, [224, 224]) / 255.0
    image = tf.expand_dims(image, axis=0)

    # Run the TFLite model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image.numpy())
    interpreter.invoke()

    # Get predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])
    category_index = np.argmax(predictions[0])
    confidence = float(predictions[0][category_index])

    categories = [
    ('WOMEN', 'Tees_Tanks'), ('WOMEN', 'Blouses_Shirts'), ('WOMEN', 'Dresses'),
    ('WOMEN', 'Skirts'), ('MEN', 'Pants'), ('WOMEN', 'Sweaters'),
    ('WOMEN', 'Shorts'), ('WOMEN', 'Sweatshirts_Hoodies'), ('WOMEN', 'Jackets_Coats'),
    ('WOMEN', 'Denim'), ('WOMEN', 'Graphic_Tees'), ('MEN', 'Tees_Tanks'),
    ('MEN', 'Suiting'), ('WOMEN', 'Pants'), ('MEN', 'Shorts'), ('MEN', 'Sweaters'),
    ('WOMEN', 'Cardigans'), ('MEN', 'Jackets_Vests'), ('WOMEN', 'Rompers_Jumpsuits'),
    ('MEN', 'Sweatshirts_Hoodies'), ('MEN', 'Shirts_Polos'), ('WOMEN', 'Leggings'),
    ('MEN', 'Denim')
    ]
    category = categories[category_index]

    # Fetch additional details
    recommendations = fetch_asos_recommendations(category, input_data.occasion)
    trend_images = fetch_trending_images()
    openai_description = generate_openai_description(category, input_data.occasion)

    return {
        "category": category,
        "confidence": confidence,
        "recommendations": recommendations,
        "trend_images": trend_images,
        "openai_description": openai_description,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)
