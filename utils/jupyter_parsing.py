import os
from datetime import datetime

def parse_uploaded_file(uploader_widget, content_dir="data/input"):
    """
    Handles FileUpload widget input and saves the uploaded image to disk.
    Supports both VS Code (tuple format) and JupyterLab (dict format).

    Args:
        uploader_widget: The FileUpload widget instance.
        content_dir: Directory where the uploaded image should be saved.

    Returns:
        str: The saved filename (not full path).

    Raises:
        ValueError: If uploader format is unsupported.
    """
    uploader_val = uploader_widget.value

    if isinstance(uploader_val, tuple):  # VS Code
        uploaded_info = uploader_val[0]
    elif isinstance(uploader_val, dict):  # JupyterLab
        uploaded_info = list(uploader_val.values())[0]
    else:
        raise ValueError("Unsupported uploader format.")

    # Save with timestamped filename
    filename = f"user_{datetime.now().strftime('%H%M%S')}.jpg"
    save_path = os.path.join(content_dir, filename)
    os.makedirs(content_dir, exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(uploaded_info["content"])

    return filename