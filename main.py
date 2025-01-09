from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import openai
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model (For demo purposes, replace with actual model load code)
# model = load_model("path_to_your_model")

class ImageURL(BaseModel):
    url: str
    occasion: str

@app.post("/predict")
async def predict(file: UploadFile = File(None), image_url: str = Form(None), occasion: str = Form(...)):
    if file:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    elif image_url:
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
    else:
        return {"error": "No image provided"}

    # Preprocess the image (implement this based on your model)
    # processed_image = preprocess_image(image)

    # Mock prediction (replace with actual prediction logic)
    prediction = "MEN-Jackets_Vests"
    confidence = 0.95

    # Fetch recommendations from ASOS API
    asos_recommendations = fetch_asos_recommendations(prediction, occasion)

    # Fetch trending images from Pexels API
    trending_images = fetch_pexels_images(prediction)

    # Generate description using OpenAI
    description = generate_openai_description(prediction, occasion)

    return {
        "category": prediction,
        "confidence": confidence,
        "recommendations": asos_recommendations,
        "trending_images": trending_images,
        "description": description
    }

def fetch_asos_recommendations(category, occasion):
    return [
        {"name": "Stylish Jacket", "image": "https://example.com/jacket.jpg", "link": "https://asos.com/jacket"},
        {"name": "Trendy Vest", "image": "https://example.com/vest.jpg", "link": "https://asos.com/vest"}
    ]

def fetch_pexels_images(category):
    api_key = os.getenv("PEXELS_API_KEY")
    url = f"https://api.pexels.com/v1/search?query={category}&per_page=5"
    headers = {"Authorization": api_key}
    response = requests.get(url, headers=headers)
    data = response.json()
    return [photo["src"]["medium"] for photo in data["photos"]]

def generate_openai_description(category, occasion):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = f"Describe fashion recommendations for {category} suitable for a {occasion} occasion."
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
