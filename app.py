import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Function to check if the image is a skin image
def is_skin_image(image):
    # Implement your logic to check if the image is a skin image
    # You can use any image processing or computer vision techniques for this task
    # For example, you can check for certain color ranges or patterns

    # Placeholder logic (replace with your own)
    width, height = image.size
    if width < 100 or height < 100:
        return False
    else:
        return True

# Function to make predictions
def predict(image):
    # Preprocess the image
    img = np.array(image)
    img = img / 255.0  # Normalize the image

    # Reshape the image for prediction
    img_reshaped = img.reshape(1, -1)

    # Make prediction
    pred = model.predict(img_reshaped)

    # Return the predicted class
    return pred[0]

# Load the trained model from the pickle file
model_file_path = 'model.sav'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit app
def main():
    st.title("Skin Cancer Classification")
    st.write("Upload an image and the model will predict whether it is benign or malignant.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)

            if is_skin_image(image):
                # st.image(image, caption="Uploaded Image", use_column_width=True)

                # Make prediction
                pred = predict(image)

                # Display prediction result
                if pred == 1:
                    st.success("Prediction: Malignant")
                else:
                    st.success("Prediction: Benign")
            else:
                st.write("Not an image of skin")

        except:
            st.write("Invalid image file")

# Run the app
if __name__ == "__main__":
    main()
