import os
from pathlib import Path

# Use CPU inference; this application does not require a GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "Handwritten_Character_Recognition.keras"

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading the trained CNN model...")
def load_model():
    """Load the trained Keras model once and cache it for later requests."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Upload Handwritten_Character_Recognition.keras to the repository root."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess_character(uploaded_image: Image.Image):
    """Convert a photo of one handwritten letter into a 28x28 model input."""
    gray_image = ImageOps.grayscale(uploaded_image)
    image_array = np.asarray(gray_image, dtype=np.uint8)

    # EMNIST samples use dark backgrounds and light strokes. Most phone photos
    # have a light paper background and dark ink, so invert those images.
    if float(image_array.mean()) > 127:
        image_array = 255 - image_array

    # Remove faint background noise and locate the written character.
    image_array = image_array.copy()
    image_array[image_array < 30] = 0
    rows, columns = np.where(image_array > 30)
    if rows.size == 0 or columns.size == 0:
        raise ValueError("No character was detected. Upload a clear handwritten letter.")

    cropped = image_array[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    character = Image.fromarray(cropped)
    character.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Center the character on a 28x28 black canvas, matching model input size.
    canvas = Image.new("L", (28, 28), color=0)
    x_offset = (28 - character.width) // 2
    y_offset = (28 - character.height) // 2
    canvas.paste(character, (x_offset, y_offset))

    model_input = np.asarray(canvas, dtype=np.float32) / 255.0
    model_input = model_input.reshape(1, 28, 28, 1)
    return model_input, canvas


st.title("✍️ Handwritten Character Recognition")
st.write(
    "Upload an image containing **one handwritten English alphabet (A–Z)**. "
    "The trained CNN model will predict the most likely character."
)
st.info(
    "For best results, write one large dark letter on plain white paper. "
    "Avoid ruled lines, shadows, and multiple characters in one image."
)

uploaded_file = st.file_uploader(
    "Upload a handwritten character image",
    type=["png", "jpg", "jpeg", "bmp"],
)

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")
        st.image(original_image, caption="Uploaded image", use_container_width=True)

        input_tensor, processed_image = preprocess_character(original_image)
        st.image(processed_image, caption="Preprocessed 28 × 28 image", width=180)

        with st.spinner("Recognizing the character..."):
            trained_model = load_model()
            probabilities = trained_model.predict(input_tensor, verbose=0)[0]

        top_indices = np.argsort(probabilities)[-3:][::-1]
        predicted_index = int(top_indices[0])
        predicted_letter = chr(predicted_index + ord("A"))
        confidence = float(probabilities[predicted_index]) * 100

        st.subheader("Prediction")
        st.metric("Predicted character", predicted_letter, f"{confidence:.2f}% confidence")

        st.subheader("Top 3 predictions")
        top_results = [
            {
                "Character": chr(int(index) + ord("A")),
                "Confidence": f"{float(probabilities[index]) * 100:.2f}%",
            }
            for index in top_indices
        ]
        st.table(top_results)
        st.caption(
            "Confidence is the model's softmax score for the predicted class; "
            "it is not a guarantee that the prediction is correct."
        )

    except (OSError, ValueError) as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"The model could not process this image: {error}")
else:
    st.caption("Choose an image above to begin.")

st.divider()
st.caption("Built with Python, TensorFlow/Keras, EMNIST Letters, and Streamlit.")
