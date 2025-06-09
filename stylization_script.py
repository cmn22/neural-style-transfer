"""
Stylize images using a pretrained transformer-based style transfer model.

Supports both single image and batch processing using a provided model checkpoint.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader

# Custom utility functions and dataset
import utils.utils as utils
from models.definitions.transformer_net import TransformerNet


def stylize_static_image(config):
    """
    Apply style transfer to images using a pretrained model.

    Args:
        config (dict): Configuration dictionary with paths and settings.
    """
    # Setup device (use GPU if available)
    if torch.backends.mps.is_available():
        device = torch.device("mps")    # for Apple Silicone
    elif torch.cuda.is_available():
        device = torch.device("cuda")   # for Nvidia
    else:
        device = torch.device("cpu")

    # Load the model and apply pretrained weights
    model = TransformerNet().to(device)
    model_path = os.path.join(config["model_binaries_path"], config["model_name"])
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if config["verbose"]:
        utils.print_model_metadata(checkpoint)

    with torch.no_grad():
        # Batch processing: directory input
        if os.path.isdir(config["content_input"]):
            dataset = utils.SimpleDataset(config["content_input"], config["img_width"])
            loader = DataLoader(dataset, batch_size=config["batch_size"])

            total = len(dataset)
            processed = 0

            try:
                for batch_idx, batch_imgs in enumerate(loader):
                    batch_imgs = batch_imgs.to(device)
                    stylized_batch = model(batch_imgs).to('cpu').numpy()

                    for img in stylized_batch:
                        utils.save_and_maybe_display_image(config, img, should_display=False)

                    processed += len(batch_imgs)
                    if config["verbose"]:
                        print(f"Processed batch {batch_idx + 1}: {processed}/{total} images")
            except Exception as e:
                print(f"Error during batch processing: {e}")
                print(f"Consider lowering batch_size={config['batch_size']} or img_width={config['img_width']}")
                exit(1)

        # Single image processing
        else:
            input_path = os.path.join(config["content_images_path"], config["content_input"])
            input_img = utils.prepare_img(input_path, config["img_width"], device)
            output_img = model(input_img).to('cpu').numpy()[0]
            utils.save_and_maybe_display_image(config, output_img, should_display=config["should_not_display"])


def build_config():
    """
    Parses CLI arguments and builds an inference configuration dictionary.

    Returns:
        dict: Complete inference configuration.
    """
    # Project-relative paths
    base_dir = os.path.dirname(__file__)
    content_path = os.path.join(base_dir, 'data', 'input')
    output_path = os.path.join(base_dir, 'data', 'output')
    model_path = os.path.join(base_dir, 'models', 'binaries')

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Argument parser for CLI input
    parser = argparse.ArgumentParser(description="Apply neural style transfer to images")

    # Core arguments
    parser.add_argument(
        "--content_input", 
        type=str, 
        default="lion.jpg",
        help="Image filename or directory to stylize (under 'data/input')"
    )

    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=5,
        help="Batch size for directory stylization"
    )

    parser.add_argument(
        "--img_width",
        type=int, 
        default=500,
        help="Resize content image(s) to this width in pixels"
    )

    parser.add_argument(
        "--model_name", 
        type=str, 
        default="vg_starry_night.pth",
        help="Pretrained model file name (under 'models/binaries')"
    )

    # Optional flags
    parser.add_argument(
        "--should_not_display", 
        action="store_false",
        help="Disable live display of stylized images"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra debug information"
    )

    parser.add_argument(
        "--redirected_output",
        type=str, 
        default=None,
        help="Custom output directory (default is 'data/output')"
    )

    args = parser.parse_args()

    # Use redirected output only if provided (useful for integration/submodules)
    output_dir = args.redirected_output or output_path

    # Build configuration dictionary
    config = {
        "content_input": args.content_input,
        "batch_size": args.batch_size,
        "img_width": args.img_width,
        "model_name": args.model_name,
        "should_not_display": args.should_not_display,
        "verbose": args.verbose,
        "redirected_output": args.redirected_output,
        "content_images_path": content_path,
        "output_images_path": output_dir,
        "model_binaries_path": model_path,
    }

    return config


if __name__ == "__main__":
    # Prepare config and run the stylization process
    config = build_config()
    stylize_static_image(config)