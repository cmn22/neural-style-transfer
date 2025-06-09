"""
utils.py - Utility functions and classes for neural style transfer project

This module includes:
- Image loading, preprocessing, and postprocessing utilities
- Dataset and DataLoader helpers for batch processing
- Model metadata and training helpers
- Utility functions for saving images with optional display
"""

import os
import re
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Any, List
import torch
import git

# === Constants for Image Normalization ===

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])

IMAGENET_MEAN_255 = np.array([123.675, 116.28, 103.53])
# Note: std neutral means std=1 to preserve scale for 0-255 normalization
IMAGENET_STD_NEUTRAL = np.array([1, 1, 1])

# === Dataset and DataLoader Helpers ===

class SimpleDataset(Dataset):
    """
    Dataset for loading images from a directory, resizing them to a target width,
    and applying normalization and tensor conversion.

    Args:
        img_dir (str): Path to the directory containing images.
        target_width (int): Target width to resize images; height is scaled to keep aspect ratio.
    """
    def __init__(self, img_dir, target_width):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]

        # Compute height corresponding to target width based on first image aspect ratio
        h, w = load_image(self.img_paths[0]).shape[:2]
        img_height = int(h * (target_width / w))
        self.target_width = target_width
        self.target_height = img_height

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = load_image(self.img_paths[idx], target_shape=(self.target_height, self.target_width))
        tensor = self.transform(img)
        return tensor

# === Image Loading and Preprocessing ===

def load_image(img_path, target_shape=None):
    """
    Load an image from disk, convert BGR to RGB, and optionally resize.

    Args:
        img_path (str): Path to the image file.
        target_shape (int or tuple or None): Target shape for resizing.
            If int, resizes width to target_shape and adjusts height proportionally.
            If tuple (height, width), resizes to exact shape.
            If None, no resizing is done.

    Returns:
        np.ndarray: Image as float32 array normalized to [0,1] range in RGB format.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f'Image path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # Convert BGR to RGB

    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:
            # Resize width, scale height proportionally
            h, w = img.shape[:2]
            new_w = target_shape
            new_h = int(h * (new_w / w))
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_CUBIC)
        else:
            # Resize exactly to target shape (height, width)
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    return img

def prepare_img(img_path, target_shape, device, batch_size=1, should_normalize=True, is_255_range=False):
    """
    Load and preprocess image for input into the style transfer model.

    Args:
        img_path (str): Path to the image file.
        target_shape (int): Target width to resize image.
        device (torch.device): Device to move tensor to (CPU or GPU).
        batch_size (int): Number of copies to repeat for batch processing.
        should_normalize (bool): Whether to normalize using ImageNet mean/std.
        is_255_range (bool): Whether to scale pixel values to [0,255].

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (batch_size, C, H, W).
    """
    img = load_image(img_path, target_shape=target_shape)

    transform_list = [transforms.ToTensor()]
    if is_255_range:
        transform_list.append(transforms.Lambda(lambda x: x.mul(255)))
    if should_normalize:
        if is_255_range:
            transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL))
        else:
            transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1))

    transform = transforms.Compose(transform_list)

    img = transform(img).to(device)
    img = img.repeat(batch_size, 1, 1, 1)  # Repeat for batch size

    return img

def post_process_image(dump_img):
    """
    Postprocess output tensor from the model to a displayable uint8 RGB image.

    Args:
        dump_img (np.ndarray): Output image tensor, shape (C, H, W), normalized.

    Returns:
        np.ndarray: uint8 RGB image with shape (H, W, C).
    """
    assert isinstance(dump_img, np.ndarray), f'Expected numpy array, got {type(dump_img)}'

    mean = IMAGENET_MEAN_1.reshape(-1, 1, 1)
    std = IMAGENET_STD_1.reshape(-1, 1, 1)
    dump_img = (dump_img * std) + mean  # De-normalize
    dump_img = (np.clip(dump_img, 0., 1.) * 255).astype(np.uint8)
    dump_img = np.moveaxis(dump_img, 0, 2)  # CHW to HWC
    return dump_img

# === Image Saving and Display ===

def get_next_available_name(input_dir):
    """
    Generate a filename with a 6-digit zero-padded index that does not exist yet.

    Args:
        input_dir (str): Directory to check existing files.

    Returns:
        str: Next available filename, e.g., '000001.jpg'.
    """
    img_name_pattern = re.compile(r'[0-9]{6}\.jpg')
    candidates = [f for f in os.listdir(input_dir) if re.fullmatch(img_name_pattern, f)]

    if not candidates:
        return '000000.jpg'
    else:
        latest_file = sorted(candidates)[-1]
        prefix_int = int(latest_file.split('.')[0])
        return f'{str(prefix_int + 1).zfill(6)}.jpg'

def save_and_maybe_display_image(inference_config, dump_img, should_display=False):
    """
    Save the stylized image to disk, optionally display it.

    Args:
        inference_config (dict): Contains output path info and naming conventions.
        dump_img (np.ndarray): Stylized image as numpy array (CHW, uint8).
        should_display (bool): If True, show the image with matplotlib.

    """
    assert isinstance(dump_img, np.ndarray), f'Expected numpy array, got {type(dump_img)}.'

    dump_img = post_process_image(dump_img)

    if inference_config.get('img_width') is None:
        inference_config['img_width'] = dump_img.shape[0]

    if inference_config.get('redirected_output') is None:
        dump_dir = inference_config['output_images_path']
        dump_img_name = (
            os.path.basename(inference_config['content_input']).split('.')[0]
            + f"_width_{inference_config['img_width']}_model_"
            + inference_config['model_name'].split('.')[0]
            + '.jpg'
        )
    else:
        dump_dir = inference_config['redirected_output']
        os.makedirs(dump_dir, exist_ok=True)
        dump_img_name = get_next_available_name(dump_dir)

    cv.imwrite(os.path.join(dump_dir, dump_img_name), dump_img[:, :, ::-1])  # Convert RGB to BGR for OpenCV

    # Print info except for batch mode (content_input as directory)
    if inference_config.get('verbose') and not os.path.isdir(inference_config['content_input']):
        print(f'Saved image to {os.path.join(dump_dir, dump_img_name)}.')

    if should_display:
        plt.imshow(dump_img)
        plt.axis('off')
        plt.show()

# === Dataset Sampling Helpers ===

class SequentialSubsetSampler(Sampler):
    """
    Samples elements sequentially from a subset of the dataset.

    Args:
        data_source (Dataset): Dataset to sample from.
        subset_size (int or None): Number of elements in subset; None means entire dataset.
    """
    def __init__(self, data_source, subset_size):
        assert isinstance(data_source, Dataset) or isinstance(data_source, datasets.ImageFolder)
        self.data_source = data_source

        if subset_size is None:
            subset_size = len(data_source)
        assert 0 < subset_size <= len(data_source), f"Subset size should be between (0, {len(data_source)}]"
        self.subset_size = subset_size

    def __iter__(self):
        return iter(range(self.subset_size))

    def __len__(self):
        return self.subset_size

# === Training Data Loader Helper ===

def get_training_data_loader(training_config, should_normalize=True, is_255_range=False):
    """
    Returns a DataLoader for training feed-forward NST models.

    Args:
        training_config (dict): Contains keys 'image_size', 'dataset_path', 'subset_size', 'batch_size', 'num_of_epochs'.
        should_normalize (bool): Normalize images during loading.
        is_255_range (bool): Whether images are in 0-255 range.

    Returns:
        DataLoader: Pytorch DataLoader for training data.
    """
    transform_list = [
        transforms.Resize(training_config['image_size']),
        transforms.CenterCrop(training_config['image_size']),
        transforms.ToTensor()
    ]

    if is_255_range:
        transform_list.append(transforms.Lambda(lambda x: x.mul(255)))

    if should_normalize:
        if is_255_range:
            transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL))
        else:
            transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1))

    transform = transforms.Compose(transform_list)

    train_dataset = datasets.ImageFolder(training_config['dataset_path'], transform)
    sampler = SequentialSubsetSampler(train_dataset, training_config['subset_size'])
    training_config['subset_size'] = len(sampler)  # update subset size in config
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], sampler=sampler, drop_last=True)

    total_data_points = len(train_loader) * training_config["batch_size"] * training_config["num_of_epochs"]
    print(f"Using {total_data_points} datapoints ({len(train_loader)*training_config['num_of_epochs']} batches) for training.")
    return train_loader

# === Utility Functions ===

def gram_matrix(x, should_normalize=True):
    """
    Compute Gram matrix for style loss.

    Args:
        x (torch.Tensor): Feature tensor of shape (B, C, H, W).
        should_normalize (bool): Whether to normalize the Gram matrix.

    Returns:
        torch.Tensor: Gram matrix of shape (B, C, C).
    """
    b, ch, h, w = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= (ch * h * w)
    return gram

def normalize_batch(batch):
    """
    Normalize a batch of images from [0,255] to ImageNet mean/std.

    Args:
        batch (torch.Tensor): Batch of images (B,C,H,W) with values in [0,255].

    Returns:
        torch.Tensor: Normalized batch.
    """
    batch = batch / 255.0
    mean = batch.new_tensor(IMAGENET_MEAN_1).view(-1,1,1)
    std = batch.new_tensor(IMAGENET_STD_1).view(-1,1,1)
    return (batch - mean) / std

def total_variation(img_batch):
    """
    Calculate total variation loss to encourage spatial smoothness.

    Args:
        img_batch (torch.Tensor): Batch of images (B,C,H,W).

    Returns:
        torch.Tensor: Scalar TV loss.
    """
    batch_size = img_batch.shape[0]
    tv_loss = (torch.sum(torch.abs(img_batch[:, :, :, :-1] - img_batch[:, :, :, 1:])) +
               torch.sum(torch.abs(img_batch[:, :, :-1, :] - img_batch[:, :, 1:, :]))) / batch_size
    return tv_loss

def print_header(training_config):
    """
    Print training configuration header with hyperparameters and settings.

    Args:
        training_config (dict): Contains training parameters and flags.
    """
    print(f"Learning the style of {training_config['style_img_name']} style image.")
    print("*" * 80)
    print(f"Hyperparams: content_weight={training_config['content_weight']}, "
          f"style_weight={training_config['style_weight']}, tv_weight={training_config['tv_weight']}")
    print("*" * 80)

    if training_config.get("console_log_freq"):
        print(f"Logging to console every {training_config['console_log_freq']} batches.")
    else:
        print("Console logging disabled. Change console_log_freq to enable.")

    if training_config.get("checkpoint_freq"):
        print(f"Saving checkpoint models every {training_config['checkpoint_freq']} batches.")
    else:
        print("Checkpoint models saving disabled.")

    if training_config.get('enable_tensorboard'):
        print("Tensorboard enabled.")
        print('Run "tensorboard --logdir=runs --samples_per_plugin images=50"')
        print("Open http://localhost:6006/ in your browser.")
    else:
        print("Tensorboard disabled.")
    print("*" * 80)

def get_training_metadata(training_config):
    """
    Collect metadata info about the training run.

    Args:
        training_config (dict): Training parameters.

    Returns:
        dict: Metadata including commit hash and hyperparameters.
    """
    num_of_datapoints = training_config['subset_size'] * training_config['num_of_epochs']
    training_metadata = {
        "commit_hash": git.Repo(search_parent_directories=True).head.object.hexsha,
        "content_weight": training_config['content_weight'],
        "style_weight": training_config['style_weight'],
        "tv_weight": training_config['tv_weight'],
        "num_of_datapoints": num_of_datapoints
    }
    return training_metadata

def print_model_metadata(training_state):
    """
    Print model training metadata.

    Args:
        training_state (dict): Model checkpoint dict containing metadata keys.
    """
    print("Model training metadata:")
    for key, value in training_state.items():
        if key not in ['state_dict', 'optimizer_state']:
            print(f"{key} : {value}")

def dir_contains_only_models(path):
    """
    Verify that a directory contains only model binary files (.pt or .pth).

    Args:
        path (str): Directory path.

    Returns:
        bool: True if only model files found, else False.

    Raises:
        AssertionError: If path does not exist or is not directory.
    """
    assert os.path.exists(path), f'Provided path: {path} does not exist.'
    assert os.path.isdir(path), f'Provided path: {path} is not a directory.'
    files = os.listdir(path)
    assert files, 'No models found in the directory.'

    for f in files:
        if not (f.endswith('.pt') or f.endswith('.pth')):
            return False
    return True

def count_parameters(model):
    """
    Count total number of trainable parameters in the model.

    Args:
        model (torch.nn.Module): PyTorch model.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# === Video Frame to Tensor Helper ===
def frame_to_tensor(frame, device, should_normalize=True):
    """
    Convert a BGR video frame (OpenCV format) to a normalized torch tensor.

    Args:
        frame (np.ndarray): Input BGR frame from OpenCV (H, W, C).
        device (torch.device): Device to move the tensor to.
        should_normalize (bool): Whether to apply ImageNet normalization.

    Returns:
        torch.Tensor: Tensor of shape (1, C, H, W) ready for model input.
    """
    img = frame[:, :, ::-1]  # Convert BGR to RGB
    img = img.astype(np.float32) / 255.0

    transform_list: List[Any] = [transforms.ToTensor()]
    if should_normalize:
        transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1))

    transform = transforms.Compose(transform_list)
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor