# video_feedforward_nst.py

import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm

from models.definitions.transformer_net import TransformerNet
import utils.utils as utils

def stylize_video(config):
    """
    Stylize a video using a pretrained feedforward NST model.

    Args:
        config (dict): Contains paths and processing parameters.
    """
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load pretrained model
    model_path = os.path.join(config["model_binaries_path"], config["model_name"])
    model = TransformerNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if config["verbose"]:
        utils.print_model_metadata(checkpoint)

    # Open input video
    cap = cv2.VideoCapture(config["input_video"])
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {config['input_video']}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(config["output_path"], exist_ok=True)
    output_path = os.path.join(config["output_path"], config["output_name"])
    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) # type: ignore

    # Frame-by-frame stylization
    prev_stylized_frame = None
    smoothing_alpha = config["smoothing_alpha"]  # Higher = more weight to previous frame
    with torch.no_grad():
        for _ in tqdm(range(total_frames), desc="Stylizing frames"):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB and preprocess
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            content_tensor = utils.frame_to_tensor(rgb_frame, device, config["img_width"])

            # Stylize
            output_tensor = model(content_tensor).cpu().numpy()[0]
            stylized_frame = utils.post_process_image(output_tensor)

            # Temporal smoothing: blend with previous stylized frame
            if prev_stylized_frame is not None:
                stylized_frame = cv2.addWeighted(stylized_frame, 1 - smoothing_alpha, prev_stylized_frame, smoothing_alpha, 0)
            prev_stylized_frame = stylized_frame.copy()

            # Write to video
            bgr_stylized = cv2.cvtColor(stylized_frame, cv2.COLOR_RGB2BGR)
            out_writer.write(bgr_stylized)

    cap.release()
    out_writer.release()
    print(f"\nStylized video saved at: {output_path}")

def build_config():
    """
    Parse CLI args and return config dict.
    """
    base_dir = os.path.dirname(__file__)
    input_path = os.path.join(base_dir, 'data', 'input')
    output_path = os.path.join(base_dir, 'data', 'output')
    model_path = os.path.join(base_dir, 'models', 'binaries')

    parser = argparse.ArgumentParser(description="Feedforward Neural Style Transfer on Video")

    parser.add_argument(
        "--input_video",
        type=str,
        default="input.mp4",
        help="Video to be stylized (under 'data/input')"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="vg_starry_night.pth",
        help="Pretrained model name (under 'models/binaries')"
    )

    parser.add_argument(
        "--img_width",
        type=int,
        default=500,
        help="Resize width for processing frames"
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="styled_output.mp4",
        help="Filename for stylized video"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print metadata and debug info"
    )

    parser.add_argument(
        "--smoothing_alpha",
        type=float,
        default=0.3,
        help="Blending factor for temporal smoothing (0.0 to 1.0)"
    )

    args = parser.parse_args()

    return {
        "input_video": os.path.join(input_path, args.input_video),
        "output_name": args.output_name,
        "img_width": args.img_width,
        "verbose": args.verbose,
        "model_name": args.model_name,
        "output_path": output_path,
        "model_binaries_path": model_path,
        "smoothing_alpha": args.smoothing_alpha,
    }

if __name__ == "__main__":
    config = build_config()
    stylize_video(config)