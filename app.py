import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from utils import set_background

# Function to load the model (optimized with st.cache_data)
@st.cache_data(ttl=86400)
def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
    return model

# Set the background
set_background("./imgs/background.png")

# Load the model
MODEL_PATH = "./model_covid_pneumonia.h5"
model = load_model(MODEL_PATH)

# Function to preprocess the image and make a prediction
def model_prediction(img):
    try:
        img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img_format = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        predict_img = np.expand_dims(img_format, axis=0)
        predict_img = np.vstack([predict_img])
        result = model.predict(predict_img)

        arg_max_result = np.argmax(result)
        per = result[0][arg_max_result] * 100

        if arg_max_result == 0:
            patient_result = f"The Patient has Covid | Percentage: {int(per)} %"
        elif arg_max_result == 1:
            patient_result = f"The Patient has a Normal X-Ray | Percentage: {int(per)} %"
        elif arg_max_result == 2:
            patient_result = f"The Patient has Viral Pneumonia | Percentage: {int(per)} %"

        return img_format, patient_result
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, "Error in prediction"

# Main app layout
def main():
    st.title("💉 Covid and Pneumonia Recognition with X-Ray 🦠")

    st.subheader("Computer Vision Project using TensorFlow and CNN 🧪")
    # st.image("./imgs/0100.jpeg", width=450)
    st.image("./imgs/Covid-19-pneumonia.png", width=450)
    st.write("The Multi Classification Model was trained with over 300 X-Ray Images, 100 Images for each class, Covid, Pneumonia, and Normal X-Ray. Using TensorFlow and a CNN Architecture.")

    with st.sidebar:
        st.write("## More Information")
        st.markdown("Learn more about Covid-19, Pneumonia, and the role of AI in healthcare [here](https://example.com).")
        st.write("## Data Privacy")
        st.write("Your data is secure. Images uploaded are not stored or used for any other purposes.")

    img = st.file_uploader("Upload an X-Ray Image:", type=["png", "jpg", "jpeg"])

    if img is not None:
        image = np.array(Image.open(img))
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)

        if st.button("Analyze X-Ray"):
            with st.spinner('Analyzing...'):
                xray_img, patient_result = model_prediction(image)
                if xray_img is not None:
                    st.image(xray_img, caption="Processed X-Ray", use_column_width=True)
                    st.success(patient_result)

if __name__ == '__main__':
    main()
