
"""ISOM5240 - Simple image-to-story app."""

import gc
import io

import numpy as np
import soundfile as sf
import streamlit as st
import torch
from PIL import Image, ImageOps
from kokoro import KModel, KPipeline
from transformers import AutoModelForCausalLM, AutoProcessor, pipeline


FLORENCE_MODEL = "microsoft/Florence-2-base"
STORY_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
VOICE = "am_michael"  # Michael - American Male


def describe_image(image):
    """Create a detailed description of the uploaded image."""
    processor = AutoProcessor.from_pretrained(FLORENCE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu").eval()

    task = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=task, images=image, return_tensors="pt")

    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"].float(),
            max_new_tokens=300,
            num_beams=3,
        )

    text = processor.batch_decode(output, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        text,
        task=task,
        image_size=(image.width, image.height),
    )

    description = result[task].strip()

    del model, processor
    gc.collect()
    return description


def generate_story(description):
    """Turn the image description into a 50-100 word children's story."""
    generator = pipeline(
        "text-generation",
        model=STORY_MODEL,
        device=-1,
        torch_dtype=torch.float32,
    )

    prompt = f"""
Write a warm, simple story for a child aged 3-10.
Use 50-100 words and about 5 short sentences.
Base the story on this image description.
Add a gentle adventure and a happy ending.
Return only the story.

Image description:
{description}
"""

    result = generator(
        prompt,
        max_new_tokens=160,
        do_sample=True,
        temperature=0.65,
        top_p=0.9,
        return_full_text=False,
    )

    story = " ".join(result[0]["generated_text"].strip().split())

    del generator
    gc.collect()

    if not 50 <= len(story.split()) <= 100:
        raise RuntimeError("The story was not between 50 and 100 words. Please try again.")

    return story


def generate_audio(story):
    """Read the story using Michael, the American male voice."""
    model = KModel(repo_id="hexgrad/Kokoro-82M").to("cpu").eval()
    narrator = KPipeline(
        lang_code="a",
        repo_id="hexgrad/Kokoro-82M",
        model=model,
    )

    audio_parts = []
    with torch.inference_mode():
        for _, _, audio in narrator(story, voice=VOICE, speed=0.95):
            if audio is not None:
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                audio_parts.append(np.asarray(audio, dtype=np.float32))

    if not audio_parts:
        raise RuntimeError("Audio could not be generated.")

    wav_file = io.BytesIO()
    sf.write(wav_file, np.concatenate(audio_parts), 24000, format="WAV")

    del narrator, model
    gc.collect()
    return wav_file.getvalue()


def main():
    st.title("📚 Magic Story Maker")
    st.write("Upload a picture to create a short story and listen to it.")

    uploaded_file = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
        st.image(image)

        if st.button("Create Story"):
            try:
                with st.spinner("Creating your story..."):
                    description = describe_image(image)
                    story = generate_story(description)
                    audio = generate_audio(story)

                st.subheader("Your Story")
                st.write(story)
                st.audio(audio, format="audio/wav")

            except Exception as error:
                st.error(f"Something went wrong: {error}")


if __name__ == "__main__":
    main()
