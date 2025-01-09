import warnings
warnings.filterwarnings("ignore", message="missing ScriptRunContext")

import streamlit as st
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import tensorflow as tf
import numpy as np
import requests
import openai
import os

# Initialize FastAPI
api = FastAPI()

# Allow CORS for frontend
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables for API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
ASOS_API_KEY = os.getenv("ASOS_API_KEY")

# Ensure all API keys are loaded
if not OPENAI_API_KEY or not PEXELS_API_KEY or not ASOS_API_KEY:
    raise ValueError("One or more API keys are missing. Check your environment variables.")

# Load your trained model
model_path = "/content/DRM.keras"  # Update this to the correct path if deploying
model = tf.keras.models.load_model(model_path)

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
    """Save the image to a local file."""
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
    # Use occasion as a filter in the API request if supported, or append it to the query
    response = requests.get(
        f"https://api.asos.com/recommendations/{category}?occasion={occasion}", 
        headers=headers
    )
    if response.status_code == 200:
        # Assuming ASOS API provides image URLs
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

# API Endpoints
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
    image = tf.image.resize(image, [224, 224]) / 255.0  # Adjust for your model
    image = tf.expand_dims(image, axis=0)

    # Make prediction
    predictions = model.predict(image)
    category_index = np.argmax(predictions[0])
    confidence = float(predictions[0][category_index])

    # Example categories
    categories = ['MEN-Jackets_Vests', 'MEN-Shirts_Polos', 'WOMEN-Tees_Tanks']

    # Recommendations, Trends, and OpenAI description
    category = categories[category_index]
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

# Streamlit Frontend
def main():
    st.title("Fashion Recommendation System with Occasion Filter")
    st.write("Upload an image or enter its URL to predict the category, view occasion-specific recommendations, trends, and descriptions.")

    # Image Input
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    image_url = st.text_input("Or enter an image URL")

    # Occasion Input
    occasion = st.selectbox(
        "Select an occasion",
        ["Casual", "Formal", "Party", "Wedding", "Vacation", "Work"]
    )

    if (uploaded_image or image_url) and occasion:
        if uploaded_image:
            # Display uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            image_data = uploaded_image.read()
        else:
            # Display image from URL
            response = requests.get(image_url)
            if response.status_code == 200:
                st.image(image_url, caption="Input Image", use_column_width=True)
                image_data = response.content
            else:
                st.error("Failed to fetch the image from the URL.")
                return

        # Call API
        st.write("Predicting...")
        predictions = predict(PredictionInput(image_url=image_url, occasion=occasion))
        category = predictions["category"]
        confidence = predictions["confidence"]
        recommendations = predictions["recommendations"]
        trend_images = predictions["trend_images"]
        openai_description = predictions["openai_description"]

        # Display Results
        st.subheader("Predicted Category")
        st.write(f"Category: {category} (Confidence: {confidence:.2f})")

        st.subheader(f"Recommendations for {occasion}")
        for item in recommendations:
            st.image(item["image"], caption=item["name"], use_column_width=True)

        st.subheader("Trending Images")
        for img in trend_images:
            st.image(img, use_column_width=True)

        st.subheader("OpenAI Description")
        st.write(openai_description)

if __name__ == "__main__":
    # Run FastAPI server in a separate thread
    import threading
    threading.Thread(target=lambda: uvicorn.run(api, host="0.0.0.0", port=8000)).start()
    # Launch Streamlit interface
    main()
