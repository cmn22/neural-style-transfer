import os
import argparse
import time

import torch
from torch.optim import Adam
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np

from models.definitions.perceptual_loss_net import PerceptualLossNet
from models.definitions.transformer_net import TransformerNet
import utils.utils as utils


def train_model(training_config):
    """
    Train the neural style transfer Transformer network.

    Args:
        training_config (dict): Configuration dictionary containing all training parameters and paths.
    """
    # Setup TensorBoard writer for logging (default output: ./runs/)
    writer = SummaryWriter()
    if torch.backends.mps.is_available():
        device = torch.device("mps")    # for Apple Silicone
    elif torch.cuda.is_available():
        device = torch.device("cuda")   # for Nvidia
    else:
        device = torch.device("cpu")

    # Prepare training data loader
    train_loader = utils.get_training_data_loader(training_config)

    # Initialize networks
    transformer_net = TransformerNet().train().to(device)
    perceptual_loss_net = PerceptualLossNet(requires_grad=False).to(device)

    # Optimizer setup
    optimizer = Adam(transformer_net.parameters())

    # Load and preprocess style image
    style_img_path = os.path.join(training_config['style_images_path'], training_config['style_img_name'])
    style_img = utils.prepare_img(
        style_img_path,
        target_shape=512,  # use original style image size
        device=device,
        batch_size=training_config['batch_size']
    )

    # Extract style features from the style image (Gram matrices)
    style_features = perceptual_loss_net(style_img)
    target_style_grams = [utils.gram_matrix(x) for x in style_features]

    # Print training header info
    utils.print_header(training_config)

    # Initialize accumulators for loss tracking
    acc_content_loss, acc_style_loss, acc_tv_loss = 0.0, 0.0, 0.0
    start_time = time.time()

    # Training loop: epochs and batches
    for epoch in range(training_config['num_of_epochs']):
        for batch_id, (content_batch, _) in enumerate(train_loader):
            content_batch = content_batch.to(device)

            # Forward pass: stylize content images
            stylized_batch = transformer_net(content_batch)

            # Extract perceptual features for content and stylized images
            content_features = perceptual_loss_net(content_batch)
            stylized_features = perceptual_loss_net(stylized_batch)

            # Compute content loss (MSE of relu2_2 feature maps)
            target_content = content_features.relu2_2
            current_content = stylized_features.relu2_2
            content_loss = training_config['content_weight'] * torch.nn.MSELoss(reduction='mean')(target_content, current_content)

            # Compute style loss (MSE between Gram matrices of style and stylized images)
            current_style_grams = [utils.gram_matrix(x) for x in stylized_features]
            style_loss = 0.0
            for target_gram, current_gram in zip(target_style_grams, current_style_grams):
                style_loss += torch.nn.MSELoss(reduction='mean')(target_gram, current_gram)
            style_loss = (style_loss / len(target_style_grams)) * training_config['style_weight']

            # Compute total variation loss to encourage smoothness
            tv_loss = training_config['tv_weight'] * utils.total_variation(stylized_batch)

            # Total loss
            total_loss = content_loss + style_loss + tv_loss

            # Backpropagation and optimizer step
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Accumulate losses for logging
            acc_content_loss += content_loss.item()
            acc_style_loss += style_loss.item()
            acc_tv_loss += tv_loss.item()

            # TensorBoard logging (scalars and images)
            if training_config['enable_tensorboard']:
                global_step = len(train_loader) * epoch + batch_id + 1
                writer.add_scalar('Loss/content', content_loss.item(), global_step)
                writer.add_scalar('Loss/style', style_loss.item(), global_step)
                writer.add_scalar('Loss/total_variation', tv_loss.item(), global_step)
                writer.add_scalars('Statistics/stylized_img_stats', {
                    'min': torch.min(stylized_batch),
                    'max': torch.max(stylized_batch),
                    'mean': torch.mean(stylized_batch),
                    'median': torch.median(stylized_batch)
                }, global_step)
                if batch_id % training_config['image_log_freq'] == 0:
                    stylized_img = utils.post_process_image(stylized_batch[0].detach().cpu().numpy())
                    stylized_img = np.moveaxis(stylized_img, 2, 0)  # Convert to C,H,W for tensorboard
                    writer.add_image('Stylized Image', stylized_img, global_step)

            # Console logging at specified frequency
            if (training_config['console_log_freq'] is not None and
                    batch_id % training_config['console_log_freq'] == 0):
                elapsed_mins = (time.time() - start_time) / 60
                avg_c_loss = acc_content_loss / training_config['console_log_freq']
                avg_s_loss = acc_style_loss / training_config['console_log_freq']
                avg_tv_loss = acc_tv_loss / training_config['console_log_freq']
                avg_total_loss = (acc_content_loss + acc_style_loss + acc_tv_loss) / training_config['console_log_freq']
                print(f'time elapsed={elapsed_mins:.2f} min | epoch={epoch+1} | batch={batch_id+1}/{len(train_loader)} | '
                      f'content_loss={avg_c_loss:.4f} | style_loss={avg_s_loss:.4f} | tv_loss={avg_tv_loss:.4f} | total_loss={avg_total_loss:.4f}')
                acc_content_loss, acc_style_loss, acc_tv_loss = 0.0, 0.0, 0.0

            # Save checkpoint model at specified frequency
            if (training_config['checkpoint_freq'] is not None and
                    (batch_id + 1) % training_config['checkpoint_freq'] == 0):
                checkpoint_state = utils.get_training_metadata(training_config)
                checkpoint_state["state_dict"] = transformer_net.state_dict()
                checkpoint_state["optimizer_state"] = optimizer.state_dict()
                ckpt_name = (f"ckpt_style_{training_config['style_img_name'].split('.')[0]}"
                             f"_cw_{training_config['content_weight']}_sw_{training_config['style_weight']}"
                             f"_tv_{training_config['tv_weight']}_epoch_{epoch}_batch_{batch_id}.pth")
                torch.save(checkpoint_state, os.path.join(training_config['checkpoints_path'], ckpt_name))

    # Save the final model with metadata
    final_state = utils.get_training_metadata(training_config)
    final_state["state_dict"] = transformer_net.state_dict()
    final_state["optimizer_state"] = optimizer.state_dict()
    final_model_name = (f"style_{training_config['style_img_name'].split('.')[0]}"
                        f"_datapoints_{final_state['num_of_datapoints']}"
                        f"_cw_{training_config['content_weight']}_sw_{training_config['style_weight']}"
                        f"_tv_{training_config['tv_weight']}.pth")
    torch.save(final_state, os.path.join(training_config['model_binaries_path'], final_model_name))


if __name__ == "__main__":
    # Fixed paths (do not change unless necessary)
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset')
    style_images_path = os.path.join(os.path.dirname(__file__), 'data', 'styles')
    model_binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
    checkpoints_root_path = os.path.join(os.path.dirname(__file__), 'models', 'checkpoints')

    # Training image size and batch size
    image_size = 256  # resize content images to this square size for training
    batch_size = 4

    assert os.path.exists(dataset_path), "MS COCO dataset missing. Please download using resource_downloader.py."

    os.makedirs(model_binaries_path, exist_ok=True)

    # Argument parsing for configurable parameters
    parser = argparse.ArgumentParser(description="Train Neural Style Transfer Transformer")
    parser.add_argument("--style_img_name", type=str, default='cubism.jpg', help="Style image filename")
    parser.add_argument("--content_weight", type=float, default=1e0, help="Weight for content loss")
    parser.add_argument("--style_weight", type=float, default=3e5, help="Weight for style loss")
    parser.add_argument("--tv_weight", type=float, default=0, help="Weight for total variation loss")
    parser.add_argument("--num_of_epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--subset_size", type=int, default=None, help="Number of images to use from dataset (None for all ~83k)")
    parser.add_argument("--enable_tensorboard", type=bool, default=True, help="Enable TensorBoard logging")
    parser.add_argument("--image_log_freq", type=int, default=100, help="Frequency (batches) to log images to TensorBoard")
    parser.add_argument("--console_log_freq", type=int, default=500, help="Frequency (batches) to print logs to console")
    parser.add_argument("--checkpoint_freq", type=int, default=2000, help="Frequency (batches) to save model checkpoints")
    args = parser.parse_args()

    # Create checkpoints directory for current style
    checkpoints_path = os.path.join(checkpoints_root_path, args.style_img_name.split('.')[0])
    if args.checkpoint_freq is not None:
        os.makedirs(checkpoints_path, exist_ok=True)

    # Build training config dictionary
    training_config = {
        **vars(args),  # all parsed args
        "dataset_path": dataset_path,
        "style_images_path": style_images_path,
        "model_binaries_path": model_binaries_path,
        "checkpoints_path": checkpoints_path,
        "image_size": image_size,
        "batch_size": batch_size,
    }

    # Start training
    train_model(training_config)