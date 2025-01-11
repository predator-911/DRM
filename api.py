import os
import tensorflow as tf
import numpy as np
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai

# Initialize FastAPI
api = FastAPI()

# Load environment variables for API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
ASOS_API_KEY = os.getenv("ASOS_API_KEY")

# Ensure all API keys are loaded
if not OPENAI_API_KEY or not PEXELS_API_KEY or not ASOS_API_KEY:
    raise ValueError("One or more API keys are missing. Check your environment variables.")

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="DRM_quantised.tflite")
interpreter.allocate_tensors()

# Define schemas
class PredictionInput(BaseModel):
    image_url: str
    occasion: str

class PredictionOutput(BaseModel):
    category: str
    confidence: float
    recommendations: list
    trend_images: list
    openai_description: str

# Helper Functions
def save_image(image_data, image_name="uploaded_image"):
    """Save the image to local file."""
    image_path = f"./{image_name}.jpg"
    with open(image_path, "wb") as f:
        f.write(image_data)
    return image_path

def fetch_trending_images():
    """Fetch trending images from Pexels."""
    headers = {"Authorization": PEXELS_API_KEY}
    response = requests.get(
        "https://api.pexels.com/v1/curated", headers=headers, params={"per_page": 5}
    )
    if response.status_code == 200:
        return [img["src"]["medium"] for img in response.json().get("photos", [])]
    return []

def fetch_asos_recommendations(category, occasion):
    """Fetch recommendations from ASOS API based on category and occasion."""
    headers = {"Authorization": ASOS_API_KEY}
    response = requests.get(
        f"https://api.asos.com/recommendations/{category}?occasion={occasion}", 
        headers=headers
    )
    if response.status_code == 200:
        items = response.json().get("items", [])
        return [{"name": item["name"], "image": item["imageUrl"]} for item in items]
    return []

def generate_openai_description(category, occasion):
    """Generate a category and occasion description using OpenAI."""
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

# API Endpoint
@api.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    # Fetch image from the URL or use the uploaded file
    if input_data.image_url:
        response = requests.get(input_data.image_url)
        if response.status_code != 200:
            return {"error": "Failed to fetch image"}
        image_data = response.content
    else:
        return {"error": "No image provided"}

    # Save the image to local storage
    image_path = save_image(image_data)

    # Preprocess image
    image = tf.image.decode_image(image_data, channels=3)
    image = tf.image.resize(image, [224, 224]) / 255.0
    image = tf.expand_dims(image, axis=0)

    # Set the input tensor for the TFLite model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor with the processed image data
    interpreter.set_tensor(input_details[0]['index'], image.numpy())

    # Run inference
    interpreter.invoke()

    # Get the prediction results
    predictions = interpreter.get_tensor(output_details[0]['index'])
    category_index = np.argmax(predictions[0])
    confidence = float(predictions[0][category_index])

    categories = ['MEN-Jackets_Vests', 'MEN-Shirts_Polos', 'WOMEN-Tees_Tanks']
    category = categories[category_index]

    # Fetch additional data
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

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)
