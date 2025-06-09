"""
Downloads the MS COCO 2014 Training Dataset zip file and extracts it to the data directory.
"""

import os
import zipfile
from torch.hub import download_url_to_file

# URL for MS COCO 2014 training images (~13 GB unzipped)
MS_COCO_2014_TRAIN_DATASET_URL = 'http://images.cocodataset.org/zips/train2014.zip'

def download_mscoco_dataset(destination_dir=None):
    """
    Downloads and extracts the MS COCO 2014 training dataset zip file.

    Args:
        destination_dir (str or None): Directory to extract dataset to.
            If None, defaults to './data/dataset'

    Returns:
        str: Path to the extracted dataset directory
    """
    if destination_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        destination_dir = os.path.join(script_dir, '..', 'data', 'dataset')
    os.makedirs(destination_dir, exist_ok=True)

    tmp_zip_path = os.path.join(destination_dir, 'train2014.zip')
    print(f"Downloading MS COCO 2014 dataset from {MS_COCO_2014_TRAIN_DATASET_URL} to {tmp_zip_path} ...")
    download_url_to_file(MS_COCO_2014_TRAIN_DATASET_URL, tmp_zip_path)

    print(f"Extracting {tmp_zip_path} to {destination_dir} ...")
    with zipfile.ZipFile(tmp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_dir)

    os.remove(tmp_zip_path)
    print(f"Removed temporary file {tmp_zip_path}")

    print(f"MS COCO dataset downloaded and extracted to {destination_dir}")
    return destination_dir

if __name__ == '__main__':
    download_mscoco_dataset()