import streamlit as st
import requests

# FastAPI URL
FASTAPI_URL = "http://localhost:8000/predict"

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

        # Prepare data for API call
        data = {
            "image_url": image_url if image_url else "",
            "occasion": occasion,
        }

        # Call FastAPI predict endpoint
        response = requests.post(FASTAPI_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            # Display Results
            st.subheader("Predicted Category")
            st.write(f"Category: {result['category']} (Confidence: {result['confidence']:.2f})")

            st.subheader(f"Recommendations for {occasion}")
            for item in result['recommendations']:
                st.image(item['image'], caption=item['name'], use_column_width=True)

            st.subheader("Trending Images")
            for img in result['trend_images']:
                st.image(img, use_column_width=True)

            st.subheader("OpenAI Description")
            st.write(result['openai_description'])
        else:
            st.error("Failed to get prediction from API.")

if __name__ == "__main__":
    main()

