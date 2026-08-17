import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

# Set page config for better layout and modern icon
st.set_page_config(
    page_title="Fish Classification AI", 
    page_icon="🐟", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }

    /* Styling for the header */
    h1 {
        color: #00d2ff !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
        margin-bottom: 0rem;
    }
    
    /* Subtitle text */
    p {
        color: #d1d5db;
        font-size: 1.1rem;
    }

    /* Glassmorphism for the sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 32, 39, 0.6) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] h2 {
        color: #00d2ff !important;
    }

    /* File uploader hover effects */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        border: 2px dashed rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease-in-out;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #00d2ff;
        background: rgba(0, 210, 255, 0.05);
        transform: translateY(-2px);
    }

    /* Success box styling */
    div[data-testid="stAlert"] {
        background: rgba(46, 204, 113, 0.1);
        border-left: 4px solid #2ecc71;
        color: #ffffff;
        border-radius: 8px;
    }

    /* Progress bar custom colors */
    .stProgress > div > div > div > div {
        background-color: #00d2ff;
        background-image: linear-gradient(90deg, #3a7bd5 0%, #00d2ff 100%);
    }

    /* Adjust image rendering */
    [data-testid="stImage"] img {
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3253/3253982.png", width=80)
    st.header("About the Project")
    st.markdown("""
    This application utilizes Deep Learning to classify images of fish into various species.
    
    **Architecture:**
    - Convolutional Neural Networks (CNN)
    - Transfer Learning (e.g. VGG16, ResNet50)
    
    **Instructions:**
    1. Upload a clear image of a fish.
    2. Wait for the AI model to process.
    3. View the prediction and confidence metrics.
    """)
    st.divider()
    st.caption("Developed for Multiclass Fish Image Classification.")

# -------------------------------------------------------------------------
# Main Application Logic
# -------------------------------------------------------------------------
st.title("🐟 Fish Image Classification")
st.write("Upload an image of a fish below, and our state-of-the-art Deep Learning model will predict its species.")

# Mocking model loading to prevent crashing if the user hasn't trained it yet
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('best_model.h5')
        return model
    except Exception as e:
        return None

model = load_model()

if model is None:
    st.warning("⚠️ **Model Not Found**: `best_model.h5` is missing. The application will run in simulation mode. Please train the model in Colab and place the `.h5` file in this directory.")

# File Uploader
uploaded_file = st.file_uploader("Drop your image here...", type=["jpg", "png", "jpeg"])

# Ensure these match the exact class indices from your Colab training generator
# This is a placeholder list; update it according to your actual dataset classes
class_names = [
    "Bangus", "Big Head Carp", "Black Sea Sprat", "Catfish", 
    "Gilt Head Bream", "Hourse Mackerel", "Red Mullet", "Red Sea Bream",
    "Salmon", "Trout"
] 

if uploaded_file is not None:
    # Create two columns for layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Fish Image', use_container_width=True)
    
    with col2:
        st.markdown("### Model Prediction")
        
        # Simulate loading time for UX
        with st.spinner('Analyzing features...'):
            time.sleep(1.5)
            
            # Preprocess the image
            img_resized = image.resize((224, 224)) 
            image_array = np.array(img_resized) / 255.0 # Rescale
            
            # Handle channels
            if len(image_array.shape) == 2: # Grayscale to RGB
                image_array = np.stack((image_array,)*3, axis=-1)
            elif image_array.shape[2] == 4: # RGBA to RGB
                image_array = image_array[:,:,:3]
                
            image_array = np.expand_dims(image_array, axis=0)
            
            if model is not None:
                # Real Prediction
                predictions = model.predict(image_array)[0]
            else:
                # Simulation Prediction (Randomized for Demo)
                np.random.seed(int(time.time()))
                predictions = np.random.dirichlet(np.ones(len(class_names)), size=1)[0]
                
            # Get the top prediction
            top_index = np.argmax(predictions)
            predicted_class = class_names[top_index]
            confidence = predictions[top_index] * 100
            
            st.success(f"**Prediction:** {predicted_class} ({confidence:.1f}%)")
            
            st.markdown("#### Confidence Scores")
            
            # Sort predictions to show top 5
            top_indices = np.argsort(predictions)[::-1][:5]
            
            for i in top_indices:
                class_name = class_names[i]
                prob = predictions[i] * 100
                
                # Use Streamlit columns for inline progress bar
                metric_col1, metric_col2 = st.columns([1, 3])
                with metric_col1:
                    st.write(f"**{class_name}**")
                with metric_col2:
                    # Render progress bar
                    st.progress(int(prob))
                    st.caption(f"{prob:.1f}%")
