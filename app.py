import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model_weather.hdf5')
    return model
def preprocess_image(image_data):
    img = Image.open(image_data)
    img = img.resize((244, 244))
    img = np.asarray(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img
def predict_weather(image_data, model):
    # Preprocess the image
    img = preprocess_image(image_data)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

weather_labels = {
    0: 'Cloudy',
    1: 'Rainy',
    2: 'Shine',
    3: 'Sunrise'}

def main():
    st.title('Weather Classifier System')
    model = load_model()
    file = st.file_uploader("Choose a weather photo from your computer", type=["jpg", "jpeg", "png"])

    if file is not None:
        image_display = Image.open(file)
        st.image(image_display, caption='Uploaded Image', use_column_width=True)

        predicted_class = predict_weather(file, model)
        predicted_label = weather_labels.get(predicted_class, 'Unknown')
        st.write(f"### Prediction: {predicted_label}")

if __name__ == '__main__':
    main()
