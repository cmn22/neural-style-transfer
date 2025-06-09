"""
Downloads pretrained style transfer models from Hugging Face Hub
"""

import os
from pathlib import Path
from typing import Dict, TypedDict
from huggingface_hub import hf_hub_download

class ModelConfig(TypedDict):
    repo_id: str
    filename: str

# Configuration
HF_REPO = "cmn22/neural_style_transfer"
MODEL_FILES: Dict[str, ModelConfig] = {
    "candy.pth": {"repo_id": HF_REPO, "filename": "candy.pth"},
    "cubism.pth": {"repo_id": HF_REPO, "filename": "cubism.pth"},
    "edtaonisl.pth": {"repo_id": HF_REPO, "filename": "edtaonisl.pth"},
    "ghibli.pth": {"repo_id": HF_REPO, "filename": "ghibli.pth"},
    "mosaic.pth": {"repo_id": HF_REPO, "filename": "mosaic.pth"},
    "pop-art.pth": {"repo_id": HF_REPO, "filename": "pop-art.pth"},
    "sci-fi.pth": {"repo_id": HF_REPO, "filename": "sci-fi.pth"},
    "vg_starry_night.pth": {"repo_id": HF_REPO, "filename": "vg_starry_night.pth"}
}

# Define model output path relative to the script location
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent / "models" / "binaries"

def download_model(model_name: str, force_redownload: bool = False) -> str:
    """
    Download a model from Hugging Face Hub

    Args:
        model_name: Key from MODEL_FILES (e.g. 'candy.pth')
        force_redownload: If True, will redownload even if file exists

    Returns:
        Path to downloaded model file

    Raises:
        ValueError: If model_name is invalid
        RuntimeError: If download fails
    """
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_FILES.keys())}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = MODELS_DIR / model_name

    if dest_path.exists() and not force_redownload:
        print(f"Model already exists: {dest_path}")
        return str(dest_path)

    try:
        print(f"Downloading {model_name}...")
        hf_hub_download(
            **MODEL_FILES[model_name],
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            force_download=force_redownload
        )
        print(f"Downloaded to {dest_path}")
        return str(dest_path)

    except Exception as e:
        raise RuntimeError(f"Failed to download {model_name}: {e}") from e

def main():
    """Download all models"""
    for name in MODEL_FILES:
        try:
            download_model(name)
        except Exception as e:
            print(f"Error with {name}: {e}")

if __name__ == "__main__":
    main()