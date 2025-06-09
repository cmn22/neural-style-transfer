import streamlit as st
import torch
import os
import tempfile
from PIL import Image
import numpy as np
from pathlib import Path

from models.definitions.transformer_net import TransformerNet
import utils.utils as utils
import utils.app_utils as app_utils
from video_nst_script import stylize_video
from image_nst_script import stylize_static_image

# --- Device Setup ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# --- Paths ---
input_dir = os.path.join("data", "input")
output_dir = os.path.join("data", "output")
model_dir = os.path.join("models", "binaries")
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# --- Page Layout ---
st.set_page_config(page_title="Neural Style Transfer", layout="wide")
st.title("Neural Style Transfer")
st.markdown(
    "<small>🔗 View the source code on <a href='https://github.com/cmn22/neural-style-transfer' target='_blank'>GitHub</a></small>",
    unsafe_allow_html=True
)
tab1, tab2 = st.tabs(["Image", "Video"])

# === Image Tab ===
with tab1:
    st.header("Image Style Transfer")

    # --- Upload or Select Content Image ---
    uploaded_file = st.file_uploader("Upload a content image (JPG or PNG)", type=["jpg", "jpeg", "png"])
    available_images = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    image_choice = st.selectbox("Or select from existing images", options=["None"] + available_images)

    # --- Model Selection ---
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".pth")])
    model_choice = st.selectbox("Choose a style model", model_files)

    # --- Resize Slider ---
    img_width = st.slider("Resize width (px)", 128, 1024, 500, step=32)

    if (uploaded_file or image_choice != "None") and model_choice:
        if st.button("Stylize Image"):
            with st.spinner("Stylizing... Please wait."):
                if uploaded_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    content_input = os.path.basename(tmp_path)
                    content_path = os.path.dirname(tmp_path)
                else:
                    content_input = image_choice
                    content_path = input_dir

                config = {
                    "content_input": content_input,
                    "batch_size": 1,
                    "img_width": img_width,
                    "model_name": model_choice,
                    "should_not_display": True,
                    "verbose": False,
                    "redirected_output": None,
                    "content_images_path": content_path,
                    "output_images_path": output_dir,
                    "model_binaries_path": model_dir,
                }

                stylized_pil = stylize_static_image(config, return_pil=True)
                st.image([os.path.join(content_path, content_input), stylized_pil], caption=["Original", "Stylized"], width=300)

                st.download_button(
                    "Download Stylized Image",
                    app_utils.pil_to_bytes(stylized_pil),
                    file_name="stylized.jpg",
                    mime="image/jpeg"
                )
    else:
        st.info("Upload a content image or choose one from folder, and select a model to begin.")

# === Video Tab ===
with tab2:
    st.header("Video Style Transfer")

    uploaded_vid = st.file_uploader("Upload a video (MP4 or MOV)", type=["mp4", "mov"])
    available_videos = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".mp4", ".mov"))])
    video_choice = st.selectbox("Or select from existing videos", options=["None"] + available_videos)

    model_choice_vid = st.selectbox("Choose a style model", model_files, key="vid_model")
    video_width = st.slider("Resize width (px)", 256, 1280, 500, step=32)
    smoothing_alpha = st.slider("Smoothing Alpha (0.0 = off)", 0.0, 1.0, 0.3, step=0.05)
    verbose = st.checkbox("Verbose Output", value=False, key="verbose_vid")

    if (uploaded_vid or video_choice != "None") and model_choice_vid:
        if st.button("Stylize Video"):
            with st.spinner("Stylizing video... This may take a while."):
                if uploaded_vid:
                    vid_path = os.path.join(input_dir, uploaded_vid.name)
                    with open(vid_path, "wb") as f:
                        f.write(uploaded_vid.getbuffer())
                    input_video = uploaded_vid.name
                else:
                    input_video = video_choice

                output_name = f"styled_{Path(input_video).stem}_{Path(model_choice_vid).stem}.mp4"

                config = {
                    "input_video": os.path.join(input_dir, input_video),
                    "output_name": output_name,
                    "img_width": video_width,
                    "verbose": verbose,
                    "model_name": model_choice_vid,
                    "output_path": output_dir,
                    "model_binaries_path": model_dir,
                    "smoothing_alpha": smoothing_alpha,
                }

                stylize_video(config)

                out_path = os.path.join(output_dir, output_name)
                st.video(out_path)

                with open(out_path, "rb") as f:
                    st.download_button(
                        "Download Stylized Video", 
                        f, file_name=output_name, 
                        mime="video/mp4"
                    )
    else:
        st.info("Upload or select a video and choose a style model to proceed.")
