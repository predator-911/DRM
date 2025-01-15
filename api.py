import requests
import tensorflow as tf
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Define your schemas
class PredictionInput(BaseModel):
    image_url: str  # URL of the image
    occasion: str    # Occasion for the clothing (e.g., "formal", "casual")

class PredictionOutput(BaseModel):
    category: str
    confidence: float
    recommendations: List[str]
    trend_images: List[str]
    openai_description: str

# Initialize the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Category list for the model (Ensure this is correct)
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

# Placeholder functions for fetching recommendations, trend images, and OpenAI description
def fetch_asos_recommendations(category, occasion):
    # Mock recommendation data
    return [
        f"Recommendation 1 for {category} during {occasion}",
        f"Recommendation 2 for {category} during {occasion}",
        f"Recommendation 3 for {category} during {occasion}"
    ]

def fetch_trending_images():
    # Mock trending image data
    return [
        "https://example.com/trending_image1.jpg",
        "https://example.com/trending_image2.jpg",
        "https://example.com/trending_image3.jpg"
    ]

def generate_openai_description(category, occasion):
    # Mock OpenAI description
    return f"Description for {category} based on {occasion}."

# FastAPI POST endpoint for prediction
@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        # Logging the input data for debugging purposes
        print(f"Received input data: {input_data}")

        # Fetch the image using the URL
        response = requests.get(input_data.image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch image")
        
        image_data = response.content

        # Preprocess the image
        image = tf.image.decode_image(image_data, channels=3)
        image = tf.image.resize(image, [224, 224]) / 255.0
        image = tf.expand_dims(image, axis=0)

        # Run the TFLite model for prediction
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], image.numpy())
        interpreter.invoke()

        # Get predictions
        predictions = interpreter.get_tensor(output_details[0]['index'])
        category_index = np.argmax(predictions[0])
        confidence = float(predictions[0][category_index])

        # Determine the predicted category
        category = categories[category_index]

        # Fetch recommendations based on the prediction and occasion
        recommendations = fetch_asos_recommendations(category, input_data.occasion)
        trend_images = fetch_trending_images()
        openai_description = generate_openai_description(category, input_data.occasion)

        # Return the predicted output
        return {
            "category": category,
            "confidence": confidence,
            "recommendations": recommendations,
            "trend_images": trend_images,
            "openai_description": openai_description,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
