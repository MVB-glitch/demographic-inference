# -*- coding: utf-8 -*-
"""Unit tests for the data_loader module."""

import os
import tempfile

import pandas as pd
import pytest

from demographic_inference.config import PipelineConfig
from demographic_inference.data_loader import is_valid, load_users_from_csv


class TestIsValid:
    def test_none(self):
        assert is_valid(None) is False

    def test_nan(self):
        assert is_valid(float("nan")) is False

    def test_empty_string(self):
        assert is_valid("") is False

    def test_na_string(self):
        assert is_valid("N/A") is False

    def test_nan_string(self):
        assert is_valid("nan") is False

    def test_image_placeholder(self):
        assert is_valid("画像") is False

    def test_none_string(self):
        assert is_valid("None") is False

    def test_valid_text(self):
        assert is_valid("Hello world") is True

    def test_valid_number(self):
        assert is_valid(42) is True

    def test_whitespace_only(self):
        assert is_valid("   ") is False


class TestLoadUsersFromCSV:
    def _create_sample_csv(self, tmp_dir: str) -> str:
        """Create a minimal sample CSV for testing."""
        data = {
            "user_id": ["u001", "u002"],
            "handle": ["@test1", "@test2"],
            "bio_description": ["東京在住のエンジニア", "N/A"],
            "profile_picture_url": ["u001_profile.jpg", "N/A"],
            "post_1_text": ["今日は渋谷に行きました", "大阪で食べたたこ焼き最高"],
            "post_1_image_url": ["u001_p1.jpg", "N/A"],
            "post_2_text": ["スカイツリーが見えた", "N/A"],
            "post_2_image_url": ["N/A", "N/A"],
            "interaction_1_bio": ["友人A: 横浜在住", "N/A"],
            "ground_truth": ["Tokyo", "Osaka"],
            "ground_truth_age": ["25-34", "35-44"],
            "ground_truth_gender": ["M", "F"],
        }
        df = pd.DataFrame(data)
        csv_path = os.path.join(tmp_dir, "test_users.csv")
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_load_basic(self, tmp_path):
        csv_path = self._create_sample_csv(str(tmp_path))
        config = PipelineConfig(max_posts=2, max_interactions=1)
        users = load_users_from_csv(csv_path, config)

        assert len(users) == 2
        assert users[0]["user_id"] == "u001"
        assert users[0]["handle"] == "@test1"
        assert len(users[0]["posts"]) == 2
        assert users[0]["posts"][0]["text"] == "今日は渋谷に行きました"
        assert users[0]["posts"][0]["image_file"] == "u001_p1.jpg"

    def test_invalid_bio_handled(self, tmp_path):
        csv_path = self._create_sample_csv(str(tmp_path))
        users = load_users_from_csv(csv_path)
        assert users[1]["bio"] == ""  # "N/A" should be treated as empty

    def test_ground_truth_loaded(self, tmp_path):
        csv_path = self._create_sample_csv(str(tmp_path))
        users = load_users_from_csv(csv_path)
        assert users[0]["ground_truth"] == "Tokyo"
        assert users[0]["ground_truth_age"] == "25-34"
        assert users[0]["ground_truth_gender"] == "M"

    def test_missing_ground_truth_columns(self, tmp_path):
        """If ground truth columns don't exist, default to 'Unknown'."""
        data = {
            "user_id": ["u001"],
            "handle": ["@test"],
            "bio_description": ["bio"],
            "profile_picture_url": ["N/A"],
            "post_1_text": ["test post"],
            "post_1_image_url": ["N/A"],
            "ground_truth": ["Tokyo"],
        }
        df = pd.DataFrame(data)
        csv_path = os.path.join(str(tmp_path), "no_demo_truth.csv")
        df.to_csv(csv_path, index=False)

        users = load_users_from_csv(csv_path)
        assert users[0]["ground_truth_age"] == "Unknown"
        assert users[0]["ground_truth_gender"] == "Unknown"
