import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from gtts import gTTS
import torch
import os

st.set_page_config(page_title="AI Storyteller", page_icon="📖")
st.title("📖 AI Storyteller — Text, Image & Voice Generator")
st.write("Enter a theme or idea, and let AI craft a story, generate a cover image, and narrate it for you!")

theme = st.text_input("Enter your story theme:", "A robot discovering emotions in the future")

if st.button("Generate Story"):
    with st.spinner("Crafting your story..."):
        # --- Generate Story ---
        story_gen = pipeline("text-generation", model="gpt2")
        story = story_gen(f"Write a short imaginative story about {theme}", max_length=300, do_sample=True)[0]['generated_text']

        st.subheader("📝 Generated Story")
        st.write(story)

        # --- Generate Image ---
        st.info("Generating story cover image...")
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        image = pipe(theme).images[0]
        image.save("story_cover.png")
        st.image(image, caption="AI-Generated Story Cover", use_container_width=True)

        # --- Generate Audio ---
        st.info("Converting story to voice...")
        tts = gTTS(text=story, lang='en')
        audio_file = "story.mp3"
        tts.save(audio_file)
        st.audio(audio_file)

        with open(audio_file, "rb") as file:
            st.download_button("🎧 Download Audio", data=file, file_name="story.mp3", mime="audio/mpeg")
