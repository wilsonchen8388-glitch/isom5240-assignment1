
import streamlit as st
from transformers import pipeline
from gtts import gTTS
from PIL import Image
import os

# Page Setup
st.set_page_config(page_title="AI Storyteller", page_icon="🎨")
st.title("🎨 AI Image Storyteller")
st.info("Upload an image and the AI will create a children's story with audio.")

# Load AI Models
@st.cache_resource
def load_models():
    # Model 1: Image Captioning
    captioner = pipeline(model="Salesforce/blip-image-captioning-base")
    # Model 2: Text Generation
    story_gen = pipeline(model="gpt2")
    return captioner, story_gen

captioner, story_gen = load_models()

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Generating your story...'):
        # Step 1: Image to Text
        raw_caption = captioner(image)[0]['generated_text']
        
        # Step 2: Generate Story (Targeted for kids 3-10)
        prompt = f"A short, happy story for children about {raw_caption}: Once upon a time"
        story_output = story_gen(prompt, max_length=100, do_sample=True, truncation=True)
        story_text = story_output[0]['generated_text']
        
        # Step 3: Text to Speech
        tts = gTTS(text=story_text, lang='en')
        tts.save("story_audio.mp3")
        
        # Display Results
        st.subheader("📖 Story")
        st.write(story_text)
        
        st.subheader("🔊 Listen")
        st.audio("story_audio.mp3")
