import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import base64  # Import base64 module for encoding

# Load the trained skin disease classification model
@st.cache(allow_output_mutation=True)
def load_skin_disease_model():
    model = load_model(r'C:\Users\mukth\OneDrive\Documents\project\skin_disease_model_ISIC_densenet.h5')
    return model

# Define the class labels
class_labels = ['Actinic keratosis',
    'Atopic Dermatitis',
    'Benign keratosis',
    'Dermatofibroma',
    'Melanocytic nevus',
    'Melanoma',
    'Squamous cell carcinoma',
    'Tinea Ringworm Candidiasis',
    'Vascular lesion']

# Define prevention suggestions for each disease
prevention_suggestions = {
    'Actinic keratosis': 'Avoid excessive sun exposure and use sunscreen regularly.',
    'Atopic Dermatitis': 'Keep your skin moisturized and avoid triggers like harsh soaps and certain fabrics.',
    'Benign keratosis': 'Regularly exfoliate and use sunscreen to prevent further growth.',
    'Dermatofibroma': 'No specific prevention measures, but monitoring changes in size or color is important.',
    'Melanocytic nevus': 'Regularly monitor for changes in size, shape, or color and protect from sun exposure.',
    'Melanoma': 'Protect your skin from excessive sun exposure and perform regular self-examinations.',
    'Squamous cell carcinoma': 'Protect your skin from sun exposure and avoid tanning beds.',
    'Tinea Ringworm Candidiasis': 'Practice good hygiene, keep skin clean and dry, and avoid sharing personal items.',
    'Vascular lesion': 'No specific prevention measures, but monitoring changes is important.'
}

# Function to preprocess an image
def preprocess_image(image):
    img = Image.open(image)
    img = img.convert('RGB')
    img = img.resize((240, 240))
    img = np.asarray(img) / 255.
    return img

# Function to make predictions
def make_preds(model, img_array):
    label_preds = model.predict(np.expand_dims(img_array, axis=0))
    # Convert predictions to labels
    label_enc = np.argmax(label_preds, axis=1)
    return class_labels[label_enc[0]], label_preds[0][label_enc[0]]

# Main function to run the Streamlit app
def main():
    # Custom CSS for styling
    custom_css = """
        <style>
            .title {
                font-size: 36px;
                font-weight: bold;
                color: #1e4ba1;
                text-align: center;
                margin-bottom: 20px;
            }
            .subtitle {
                font-size: 24px;
                font-weight: bold;
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .image-container {
                text-align: center;
                margin-bottom: 30px;
            }
            .uploaded-image {
                border: 2px solid #1e4ba1;
                border-radius: 5px;
                max-width: 100%;
            }
            .result-container {
                text-align: center;
                margin-top: 30px;
            }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Title and subtitle
    st.markdown('<h1 class="title">DERM AI</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="subtitle">Skin Disease Classification</h2>', unsafe_allow_html=True)

    st.write('Upload an image of a skin condition for classification.')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        model = load_skin_disease_model()
        img_array = preprocess_image(uploaded_file)
        predicted_disease, confidence = make_preds(model, img_array)
        
        st.success(f'Disease: {predicted_disease} (Confidence: {confidence:.2f})')
        
        # Display prevention suggestion
        if predicted_disease != "Unknown":
            suggestion = prevention_suggestions.get(predicted_disease, "Prevention information not available.")
            st.info(f'Prevention Suggestion: {suggestion}')

        # Visualize prediction probabilities
        st.subheader('Prediction Probabilities')
        fig, ax = plt.subplots()
        ax.barh(class_labels, model.predict(np.expand_dims(img_array, axis=0))[0])
        ax.set_xlabel('Probability')
        ax.set_ylabel('Disease')
        st.pyplot(fig)

if __name__ == '__main__':
    main()
