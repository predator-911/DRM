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
    return FileResponse("favicon.ico")

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure the key is available
if not OPENAI_API_KEY:
    raise ValueError("API key is missing. Check your environment variables.")

# Load the TensorFlow Lite model
model_path = os.path.abspath("DRM_quantized.tflite")  # Use the correct model name
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Define request and response schemas
class PredictionInput(BaseModel):
    image_url: str
    occasion: str

class PredictionOutput(BaseModel):
    category: str
    confidence: float
    openai_image: str
    openai_description: str

# Helper function to generate fashion-forward images using OpenAI's DALL·E
def generate_openai_image(category, occasion):
    openai.api_key = OPENAI_API_KEY
    prompt = f"A stylish and fashionable {category} outfit for a {occasion}. The design should be modern and appealing."
    
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']
        return image_url
    except Exception as e:
        return f"Error generating image: {e}"

# Helper function to generate descriptions using OpenAI's GPT-3
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

    # Generate OpenAI content
    openai_description = generate_openai_description(category, input_data.occasion)
    openai_image = generate_openai_image(category, input_data.occasion)

    return {
        "category": category,
        "confidence": confidence,
        "openai_image": openai_image,
        "openai_description": openai_description,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)
