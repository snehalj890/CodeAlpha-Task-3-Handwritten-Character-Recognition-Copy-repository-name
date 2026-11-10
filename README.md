# Handwritten Character Recognition Using CNN

A handwritten English alphabet recognition application built with a Convolutional Neural Network (CNN), the EMNIST Letters dataset, TensorFlow/Keras, and Streamlit.

## Features

- Accepts an image containing one handwritten English letter (A–Z).
- Preprocesses the uploaded image into a 28 × 28 grayscale input.
- Uses a trained CNN model to predict the character.
- Displays the predicted character and the top three class scores.

## Project files

```text
.
├── streamlit_app.py
├── Handwritten_Character_Recognition.keras
├── Handwritten_Character_Recognition.ipynb
├── requirements.txt
└── README.md
```

The trained model file `Handwritten_Character_Recognition.keras` must be present in the repository root alongside `streamlit_app.py`.

## Run locally

1. Install Python in a supported version compatible with the packages in `requirements.txt`.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   streamlit run streamlit_app.py
   ```

## Dataset and model

- Dataset: EMNIST Letters.
- Input image size: 28 × 28 pixels, grayscale, one channel.
- Number of output classes: 26 (A–Z).
- Model: Convolutional Neural Network (CNN) implemented with TensorFlow/Keras.

## Usage

Upload a clear image containing one handwritten alphabet. Plain white paper and dark ink generally produce better results than ruled paper or images containing multiple letters.

The displayed confidence is the model's softmax score, not a guarantee of correctness.

## Notes

- The notebook contains the development and training workflow.
- The `.keras` file contains the trained model used for inference.
- Add actual test accuracy and other measured evaluation results to this README only after verifying them from the notebook's test evaluation output.
