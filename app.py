import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Class labels for the prediction
class_labels = ["weed", "non-weed"]

# Function to preprocess the input image
def preprocess_image(image):
    # Convert image to RGB
    img = image.convert('RGB')
    # Resize the image to match model input size
    img = img.resize((128, 128))
    img = np.array(img)
    img = img / 255.0  # Normalize the pixel values to the range [0, 1]
    print("preprocess image")
    return img[np.newaxis, ...]

# Function to load the model
def load_model(model_path):
    print("preprocess image")

    return tf.keras.models.load_model(model_path)

# Function to perform inference
def perform_inference(model, image):
    input_image = preprocess_image(image)
    predictions = model.predict(input_image)
    print("preprocess image")

    return predictions

# Function to display the results for a single image prediction
def display_results(prediction):
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]

    st.write("Predicted Class:", predicted_class)
    st.write("Prediction Probabilities:", prediction)

# Streamlit app
def main():
    st.title("Weed Detection App")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the model
        model_path = "sos.h5"  # Ensure this path is correct relative to where your Streamlit app runs
        model = load_model(model_path)

        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform inference and display results
        predictions = perform_inference(model, image)
        
        # Display raw prediction probabilities for debugging
        st.write("Raw Prediction Probabilities:", predictions[0])

        display_results(predictions[0])

if __name__ == "__main__":
    main()

