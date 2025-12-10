# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Download the wide baseline variant of TartanAirV2 (TAv2) dataset used in UFM (https://uniflowmatch.github.io/).
Extract the images, depth, poses and calibration corresponding to the different unqiue environments from the h5s.
"""

import argparse
import concurrent.futures
import logging
import os
import re

import numpy as np
import pandas as pd
import urllib3
from minio import Minio
from minio.error import S3Error
from PIL import Image
from tqdm import tqdm as tqdm_bar  # 🔧 FIX: Renamed import
import h5py

def download_file(client, bucket_name, obj, destination_folder):
    "Download a file from MinIO server"
    object_name = os.path.basename(obj.object_name)
    destination_file = os.path.join(destination_folder, object_name)
    if not os.path.exists(destination_file):
        os.makedirs(os.path.dirname(destination_file), exist_ok=True)
        try:
            client.fget_object(bucket_name, obj.object_name, destination_file)
            logging.info(f"Download successful: {object_name}")
        except S3Error as e:
            logging.error(f"Error downloading {object_name}: {e}")
            return
    else:
        logging.info(f"File {destination_file} already exists. Skipping...")


def download_folder(folder_name, bucket_name, client, destination_folder, num_workers):
    "Download a folder from MinIO server"
    objects = list(client.list_objects(bucket_name, prefix=folder_name, recursive=True))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(download_file, client, bucket_name, obj, destination_folder)
            for obj in objects
        ]
        for future in tqdm_bar(  # 🔧 FIX: Use renamed import
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Downloading {folder_name}",
        ):
            future.result()


def download_tav2_wb(args, num_workers):
    """Download the TAv2 wide baseline dataset.

    Args:
        args: Parsed command line arguments
        num_workers: Number of workers for parallel download
    """
    # Create root directory if it doesn't exist
    os.makedirs(args.root_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        filename=os.path.join(args.root_dir, "tav2_download.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Define download client
    access_key = "jH8UFqt4oli1lmGabHeT"
    secret_key = "SU5aUyahsXB7AlgbSEYNeshm8GL2P5iKatd6iRrt"
    http_client = urllib3.PoolManager(
        cert_reqs="CERT_NONE",  # Disable SSL certificate verification
        maxsize=20,
    )
    urllib3.disable_warnings(
        urllib3.exceptions.InsecureRequestWarning
    )  # Disable SSL warnings
    client = Minio(
        "128.237.74.10:9000",
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
        http_client=http_client,
    )
    bucket_name = "tav2"

    # Define the target download directory
    target_dir = os.path.join(args.root_dir, "tav2_wb_h5")
    os.makedirs(target_dir, exist_ok=True)

    # Define mapping of folders that need to be downloaded
    download_mapping = [
        (
            "TartanAir/assembled/tartanair_640_mega_training_0203_pinhole_good/train_camera_data/",
            os.path.join(target_dir, "train_camera_data"),
        ),
        (
            "TartanAir/assembled/tartanair_640_mega_training_0203_pinhole_good/validation_camera_data/",
            os.path.join(target_dir, "val_camera_data"),
        ),
        (
            "TartanAir/assembled/tartanair_640_pinhole_test_good_imgdep/train_camera_data/",
            os.path.join(target_dir, "test_camera_data"),
        ),
        (
            "TartanAir/assembled/tartanair_640_mega_training_0203_pinhole_good/validation/",
            os.path.join(target_dir, "val"),
        ),
        (
            "TartanAir/assembled/tartanair_640_pinhole_test_good_imgdep/train/",
            os.path.join(target_dir, "test"),
        ),
        (
            "TartanAir/assembled/tartanair_640_mega_training_0203_pinhole_good/train/",
            os.path.join(target_dir, "train"),
        ),
    ]

    # Loop over the folders and download them
    for curr_download_mapping in tqdm_bar(download_mapping):  # 🔧 FIX: Use renamed import
        source_folder_name, destination_folder = curr_download_mapping
        os.makedirs(destination_folder, exist_ok=True)
        download_folder(
            source_folder_name, bucket_name, client, destination_folder, num_workers
        )


def setup_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Download TAv2 Wide Baseline Dataset (UFM Version)"
    )
    parser.add_argument(
        "-r",
        "--root_dir",
        type=str,
        required=True,
        help="Root directory for download, tav2_wb_h5 will be created in this directory",
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        default=20,
        help="Number of parallel workers for downloading",
    )
    return parser


if __name__ == "__main__":
    # Parse command line arguments
    parser = setup_parser()
    args = parser.parse_args()

    print("[INFO] Starting TartanAirV2 Wide Baseline download...")
    print(f"[INFO] Target directory: {args.root_dir}")
    print(f"[INFO] Number of workers: {args.num_workers}")
    print()
    
    # Install required packages if needed
    print("[CHECK] Checking dependencies...")
    try:
        import minio
        import pandas
        import PIL
        print("[OK] All dependencies installed")
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print()
        print("Please install required packages:")
        print("  pip install minio pandas pillow tqdm h5py")
        exit(1)
    
    # Download the h5 dataset
    download_tav2_wb(args, num_workers=args.num_workers)
    
    print()
    print("[SUCCESS] Download completed!")
    print(f"[INFO] Data saved to: {os.path.join(args.root_dir, 'tav2_wb_h5')}")