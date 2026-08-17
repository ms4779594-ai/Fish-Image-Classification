import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page config for better layout
st.set_page_config(page_title="Fish Classification", layout="wide")

st.title("Multiclass Fish Image Classification")
st.write("Upload an image of a fish and the model will predict its species.")

# Load model (placeholder)
@st.cache_resource
def load_model():
    # Replace 'best_model.h5' with your actual model file after training
    try:
        model = tf.keras.models.load_model('best_model.h5')
        return model
    except Exception as e:
        st.warning("Model not found. Please ensure 'best_model.h5' is in the current directory.")
        return None

model = load_model()

# Class labels (update these based on your dataset)
class_names = ["Species_1", "Species_2", "Species_3", "Species_4"]

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Fish Image', use_column_width=True)
    
    st.write("Classifying...")
    
    # Preprocess the image
    # Note: Target size should match your model's expected input shape (e.g., 224x224)
    image = image.resize((224, 224)) 
    image_array = np.array(image) / 255.0 # Rescale to [0, 1] as defined in project doc
    
    # Check if image has 3 channels (RGB)
    if len(image_array.shape) == 2: # Grayscale
        image_array = np.stack((image_array,)*3, axis=-1)
    elif image_array.shape[2] == 4: # RGBA
        image_array = image_array[:,:,:3]
        
    image_array = np.expand_dims(image_array, axis=0) # Add batch dimension
    
    if model is not None:
        # Make prediction
        predictions = model.predict(image_array)
        score = tf.nn.softmax(predictions[0])
        
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        st.success(f"This image most likely belongs to **{predicted_class}** with a {confidence:.2f}% confidence.")
        
        # Display all probabilities
        st.write("### Prediction Probabilities:")
        for i, class_name in enumerate(class_names):
            st.write(f"- {class_name}: {predictions[0][i] * 100:.2f}%")
