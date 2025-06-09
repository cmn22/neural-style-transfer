# 🖼️ Neural Style Transfer with Feedforward Transformer Network

This project is a complete implementation of Neural Style Transfer (NST) using a pretrained transformer network, enabling both image and video stylization using a fast, real-time feedforward model.

It includes:

- Style transfer for **single images** and **videos**
- A **Streamlit-based GUI** for a user-friendly experience
- CLI-based **scripted support** for training and inference pipelines
- Configurable settings like image width, temporal smoothing, and batch processing
- Downloaders for pretrained models and training datasets

---

## 🚀 Features

| Feature           | Description                                                |
|------------------|------------------------------------------------------------|
| **Image NST**     | Upload or choose images, apply artistic styles using a fast transformer net |
| **Video NST**     | Upload or choose videos, with optional temporal smoothing  |
| **Streamlit UI**  | Intuitive web UI for both image and video stylization     |
| **CLI Support**   | Script-based style transfer using configurable arguments   |
| **Custom Model Training** | Train your own models using MS-COCO or any dataset |

---

## 📁 Project Structure

```
.
├── app.py                          # Streamlit GUI application
├── image_nst_script.py             # Script for stylizing images
├── video_nst_script.py             # Script for stylizing videos
├── model_training_script.py        # Model training entrypoint
├── models/
│   ├── definitions/
│   │   ├── transformer_net.py         # Transformer feedforward network
│   │   └── perceptual_loss_net.py     # VGG16-based perceptual loss extractor
│   └── binaries/                    # Pretrained .pth models
├── utils/
│   ├── utils.py                       # Shared preprocessing, postprocessing, I/O, and dataset utils
│   ├── app_utils.py                   # Utility helpers for Streamlit app
│   ├── pretrained_models_downloader.py  # Script to download pre-trained style models
│   ├── training_dataset_downloader.py   # Script to download and extract COCO dataset
└── data/
    ├── input/                        # Input images and videos
    └── output/                       # Stylized results
```

---

## 📦 Installation

```bash
git clone https://github.com/your-username/neural-style-transfer.git
cd neural-style-transfer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📥 Pretrained Model Downloader

This must be run before using the GUI or CLI to stylize:

```bash
python utils/pretrained_models_downloader.py
```

This will download pretrained `.pth` files and place them in `models/binaries/`.

---

## 📦 Dataset Downloader (OPTIONAL: for Training only)

To train your own style model, download the MS-COCO dataset:

```bash
python utils/training_dataset_downloader.py
```

This downloads and extracts the COCO dataset under `data/train/`.

---

## 🖼️ Streamlit GUI Usage

```bash
streamlit run app.py
```

- **Image Tab**: Upload or select an image, choose a model, apply style, and download result.
- **Video Tab**: Upload or select a video, choose a model, optionally tune smoothing, and download result.

---

## 🧠 Model Training

Train your own model with a content-style dataset:

```bash
python model_training_script.py --dataset_path ./data/train --style_image ./styles/starry_night.jpg        --epochs 2 --batch_size 4 --style_weight 5e5 --content_weight 1e0
```

---

## 🧪 Script Usage (No GUI)

### Image Stylization (Single / Batch)

```bash
python image_nst_script.py --content_input lion.jpg --model_name mosaic.pth --img_width 512
```

### Video Stylization

```bash
python video_nst_script.py --input_video sample.mp4 --model_name mosaic.pth --img_width 500 --smoothing_alpha 0.3
```

---

## 📚 Code Walkthrough

### ✅ `app.py`

- Streamlit GUI with two tabs: Image and Video
- For image:
  - Uses `stylize_static_image(config, return_pil=True)` and shows original + styled image
- For video:
  - Uses `stylize_video(config)` and applies frame-wise style with smoothing

### ✅ `image_nst_script.py`

- Defines `stylize_static_image(config, return_pil=False)`
- Loads model, processes either:
  - A **single image** (returns PIL optionally)
  - A **directory** (batch image processing)

### ✅ `video_nst_script.py`

- Frame-by-frame video processing using OpenCV
- Applies style using `TransformerNet`
- Uses `cv2.addWeighted()` if smoothing is enabled
- Saves stylized video

### ✅ `model_training_script.py`

- Loads COCO dataset and chosen style image
- Computes perceptual loss using VGG
- Optimizes `TransformerNet`
- Supports live TensorBoard logs

### ✅ `transformer_net.py`

- Feedforward CNN
- Structure:
  - Conv → IN → ReLU
  - 5 Residual Blocks
  - Upsample + Conv + IN + ReLU
- Outputs stylized image in one pass

### ✅ `perceptual_loss_net.py`

- Loads pretrained VGG16 from torchvision
- Extracts intermediate features (e.g., relu1_2, relu2_2, relu3_3) for loss computation

### ✅ `utils/utils.py`

Core helpers:
- `prepare_img(path, width, device)` → tensor
- `post_process_image(tensor)` → RGB image
- `save_and_maybe_display_image(config, img)` → save logic
- `SimpleDataset` → supports batch image processing
- `frame_to_tensor()` and `tensor_to_frame()` for video

### ✅ `app_utils.py`

- `pil_to_bytes(pil_image)` → converts PIL object for Streamlit download

### ✅ `pretrained_models_downloader.py`

- Downloads multiple pretrained `.pth` style models from known URLs
- Saves them into `models/binaries/`
- Mandatory before GUI or scripts can be run

### ✅ `training_dataset_downloader.py`

- Downloads and unzips MS-COCO dataset
- Extracts `train2014.zip` into `data/train/train2014/`

---

## 🎨 Example Models

| Model File          | Style                    |
|---------------------|--------------------------|
| `vg_starry_night.pth` | Vincent van Gogh’s Starry Night |
| `la_muse.pth`         | Pablo Picasso’s La Muse  |
| `candy.pth`           | Bright pastel stroke style |

> Place these inside: `models/binaries/`

---

## 📷 Sample Output

| Input Image      | Style        | Output             |
|------------------|--------------|--------------------|
| Uploaded image   | Starry Night | Stylized version   |

---

## 🧪 Test It Out

Instead of running the full app immediately, you can explore the project using the interactive Jupyter notebooks:
- `General_NST_Notebook.ipynb`: explains and implements Johnson's Fast Neural Style Transfer using PyTorch
- `Image_NST_Notebook.ipynb`: demonstrates neural style transfer on **images** using a feedforward Transformer network
- `Video_NST_Notebook.ipynb`: applies a feedforward neural style transfer model to a **video**
- `NST_Model_Training_Notebook.ipynb`: demonstrates how to **train a Transformer network** for fast neural style transfer

---

## 📝 To-Do / Suggestions

- [ ] Add batch image GUI support
- [ ] Utlize Temporal Aware Networks instead of the current FastFeed Model for video stylization
- [ ] Add Semantic Segmentation feature for videos

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

- Based on [Perceptual Losses for Real-Time Style Transfer](https://arxiv.org/abs/1603.08155)
- Uses `torchvision.models.vgg16` for perceptual loss
- Portions of code and implementation adapted and inspired by [Aleksa Gordić](https://github.com/gordicaleksa) from his excellent repository:  
  [gordicaleksa/pytorch-neural-style-transfer-johnson](https://github.com/gordicaleksa/pytorch-neural-style-transfer-johnson)

---

## 🧑‍💻 Maintainer

**Chaitanya Malani**  
Email: contact@chaitanymalani.com

---

## 🛠️ GitHub Badges (Optional)

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-orange)
![Model](https://img.shields.io/badge/model-TransformerNet-purple)