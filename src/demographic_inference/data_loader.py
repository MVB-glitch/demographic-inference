# -*- coding: utf-8 -*-
"""
Data loading and preprocessing for the Demographic Inference Pipeline.

Handles CSV ingestion, user data extraction, and image loading.
"""

import os
from typing import Optional

import pandas as pd
from PIL import Image

from .config import PipelineConfig


def is_valid(val) -> bool:
    """Check if a pandas value is valid (not NaN, null, empty, or placeholder)."""
    if pd.isna(val):
        return False
    s = str(val).strip()
    return s not in ("", "N/A", "nan", "画像", "None")


def load_local_image(
    filename: Optional[str],
    image_dir: str,
    max_size: tuple = (512, 512),
) -> Optional[Image.Image]:
    """Load an image from local filesystem and resize it."""
    if not filename or not is_valid(filename):
        return None

    file_path = os.path.join(image_dir, str(filename).strip())

    try:
        img = Image.open(file_path)
        img.thumbnail(max_size)
        return img
    except FileNotFoundError:
        print(f"    [警告] 画像が見つかりません: {filename}")
        return None
    except Exception as e:
        print(f"    [警告] 画像を開けませんでした: {filename} ({e})")
        return None


def load_users_from_csv(
    csv_path: str,
    config: Optional[PipelineConfig] = None,
) -> list[dict]:
    """
    Load user data from a CSV file.

    Expected CSV columns:
    - user_id, handle, bio_description, profile_picture_url
    - post_{i}_text, post_{i}_image_url (for i in 1..max_posts)
    - interaction_{i}_bio (for i in 1..max_interactions)
    - ground_truth, ground_truth_age (optional), ground_truth_gender (optional)
    """
    if config is None:
        config = PipelineConfig()

    df = pd.read_csv(csv_path, header=0)
    df.columns = [str(col).strip() for col in df.columns]

    users_data = []
    for _, row in df.iterrows():
        # Extract posts
        posts = []
        for i in range(1, config.max_posts + 1):
            text_col = f"post_{i}_text"
            image_col = f"post_{i}_image_url"
            if text_col in df.columns and is_valid(row.get(text_col)):
                img_filename = None
                if image_col in df.columns and is_valid(row.get(image_col)):
                    img_filename = str(row[image_col])
                posts.append({"text": str(row[text_col]), "image_file": img_filename})

        # Extract interaction bios
        interactions = []
        for i in range(1, config.max_interactions + 1):
            bio_col = f"interaction_{i}_bio"
            if bio_col in df.columns and is_valid(row.get(bio_col)):
                interactions.append(str(row[bio_col]))

        user = {
            "user_id": str(row["user_id"]),
            "handle": str(row["handle"]) if is_valid(row.get("handle")) else "",
            "bio": str(row["bio_description"]) if is_valid(row.get("bio_description")) else "",
            "profile_picture_url": str(row["profile_picture_url"]) if is_valid(row.get("profile_picture_url")) else None,
            "posts": posts,
            "interactions": interactions,
            "ground_truth": str(row.get("ground_truth", "Unknown")),
            "ground_truth_age": str(row["ground_truth_age"]) if "ground_truth_age" in df.columns and is_valid(row.get("ground_truth_age")) else "Unknown",
            "ground_truth_gender": str(row["ground_truth_gender"]) if "ground_truth_gender" in df.columns and is_valid(row.get("ground_truth_gender")) else "Unknown",
        }
        users_data.append(user)

    return users_data
