# -*- coding: utf-8 -*-
"""
Configuration management for the Demographic Inference Pipeline.

Handles API key resolution, model configuration, and runtime settings.
API key resolution order: CLI argument → env var → .env file → Colab secrets → interactive prompt.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """Configuration for the inference pipeline."""

    # --- API Settings ---
    api_key: Optional[str] = None
    model_name: str = "gemini-2.5-flash"

    # --- Data Paths ---
    csv_path: Optional[str] = None
    image_dir: Optional[str] = None
    output_path: str = "results.xlsx"
    output_format: str = "xlsx"  # xlsx, csv, json

    # --- Processing ---
    max_posts: int = 15
    max_interactions: int = 5
    user_limit: Optional[int] = None
    batch_size: int = 5
    checkpoint_dir: str = "checkpoints"

    # --- Image Processing ---
    image_max_size: tuple = (512, 512)

    # --- Retry Settings ---
    max_retries: int = 3
    quota_wait_seconds: int = 60
    error_wait_seconds: int = 15
    rate_limit_delay: float = 6.0  # seconds between API calls

    # --- Display ---
    verbose: bool = False

    # --- Ground Truth ---
    kanto_prefectures: list = field(default_factory=lambda: [
        "Tokyo", "Saitama", "Chiba", "Kanagawa",
        "Ibaraki", "Tochigi", "Gunma"
    ])


def resolve_api_key(cli_key: Optional[str] = None) -> str:
    """
    Resolve the Gemini API key from multiple sources in priority order:
    1. CLI argument
    2. GEMINI_API_KEY environment variable
    3. .env file
    4. Google Colab secrets (if running in Colab)
    5. Interactive prompt (last resort)

    Args:
        cli_key: API key passed directly (e.g., from CLI argument).

    Returns:
        The resolved API key string.

    Raises:
        SystemExit: If no API key can be found and user cancels the prompt.
    """
    # 1. CLI argument
    if cli_key:
        return cli_key

    # 2. Environment variable
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key

    # 3. .env file
    try:
        from dotenv import load_dotenv
        # Search for .env in current directory and parent directories
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
            env_key = os.environ.get("GEMINI_API_KEY")
            if env_key:
                return env_key
    except ImportError:
        pass

    # 4. Google Colab secrets
    try:
        from google.colab import userdata
        raw_api_key = userdata.get("GEMINI_API_KEY")
        if isinstance(raw_api_key, dict) and "data" in raw_api_key:
            payload = raw_api_key.get("data", {}).get("payload")
            if payload:
                return payload
        elif isinstance(raw_api_key, str) and raw_api_key.strip():
            return raw_api_key
    except (ImportError, ModuleNotFoundError):
        pass
    except Exception:
        pass

    # 5. Interactive prompt
    print("⚠️  No API key found in environment or .env file.")
    print("   Set GEMINI_API_KEY in your environment or create a .env file.")
    try:
        user_key = input("Enter your GEMINI_API_KEY (or Ctrl+C to cancel): ").strip()
        if user_key:
            return user_key
    except (KeyboardInterrupt, EOFError):
        pass

    print("❌ No API key provided. Exiting.")
    raise SystemExit(1)
