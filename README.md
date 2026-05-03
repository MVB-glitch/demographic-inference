# Demographic Inference Pipeline

**Hybrid LLM + Dempster-Shafer Theory pipeline** for inferring demographics (age, gender, location) from X (Twitter) user data.

## Overview

This pipeline combines the natural language understanding capabilities of Google Gemini (LLM) with the mathematical rigor of Dempster-Shafer Theory (DST) to infer user demographics from social media data.

### Architecture

```
┌─────────────────────┐
│   Data Ingestion    │  CSV + Images
│   (data_loader.py)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Location Evidence  │     │ Demographics Evidence│
│  (llm_extractor.py) │     │  (llm_extractor.py)  │
│                     │     │                      │
│  4 independent      │     │  Single LLM call     │
│  mass functions:    │     │  → mass_age           │
│  A: Posts           │     │  → mass_gender        │
│  B: Friends/Network │     │                      │
│  C: Bio/Handle      │     └──────────┬───────────┘
│  D: Images          │                │
└─────────┬───────────┘                │
          │                            │
          ▼                            │
┌─────────────────────┐                │
│   DST Fusion        │                │
│   (dst_engine.py)   │                │
│                     │                │
│   Discount → Fuse   │                │
│   → Decide          │                │
└─────────┬───────────┘                │
          │                            │
          ▼                            ▼
┌─────────────────────────────────────────┐
│          Results & Metrics              │
│          (metrics.py)                   │
│                                         │
│  Location: Kanto / Non-Kanto / 不確実    │
│  Age: 18-24 / 25-34 / ... / 65+        │
│  Gender: 男性 / 女性 / 不確実            │
└─────────────────────────────────────────┘
```

### How the DST-LLM Hybrid Works

1. **Evidence Extraction**: The LLM analyzes 4 independent data sources and outputs mass functions `m(K)`, `m(NK)`, `m(Ω)` for each, along with a confidence score.
2. **Discounting**: Each mass function is discounted by its confidence score (Shafer's discount operation), shifting unreliable evidence toward uncertainty.
3. **Fusion**: Dempster's Rule of Combination sequentially fuses all 4 discounted masses into a single belief structure.
4. **Decision**: The hypothesis with the highest mass wins, unless Omega (uncertainty) exceeds 50%, in which case the prediction is "uncertain."

## Installation

### Prerequisites
- Python 3.10+
- A Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/demographic-inference.git
cd demographic-inference

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Or install as a package (enables the `demoinfer` command)
pip install -e .

# Create .env file
New-Item .env -ItemType File

# Open it in Notepad
notepad .env

# Add your API key in this format, then save and close:
# GEMINI_API_KEY=your_api_key_here
```

## Usage

### Command Line

```bash
# Basic usage
demoinfer --csv data/users.csv --images data/images/

# Process first 10 users in batches of 5
demoinfer --csv data/users.csv --images data/images/ --limit 10 --batch-size 5

# Export as JSON
demoinfer --csv data/users.csv --format json --output results.json

# Start fresh (ignore checkpoint)
demoinfer --csv data/users.csv --no-resume

# Using python -m
python -m demographic_inference --csv data/users.csv --images data/images/
```

### Full Options

```
Options:
  --csv CSV             Path to input CSV file (required)
  --images PATH         Path to image directory
  --output PATH         Output file path (default: results.xlsx)
  --format FORMAT       xlsx, csv, or json (default: xlsx)
  --limit N             Process only first N users
  --batch-size N        Users per checkpoint batch (default: 5)
  --max-posts N         Max posts per user (default: 15)
  --api-key KEY         Gemini API key (or set GEMINI_API_KEY env var)
  --model NAME          Gemini model (default: gemini-2.5-flash)
  --checkpoint-dir DIR  Checkpoint directory (default: checkpoints/)
  --no-resume           Start fresh, ignore existing checkpoint
  --verbose             Verbose output
```

### Google Colab

The pipeline can still be used in Colab. Set your API key in Colab secrets as `GEMINI_API_KEY` and the config module will pick it up automatically.

### CSV Format

Your input CSV should have the following columns:

| Column | Description |
|--------|-------------|
| `user_id` | Unique user identifier |
| `handle` | X (Twitter) handle |
| `bio_description` | User's bio text |
| `profile_picture_url` | Profile image filename |
| `post_{1..15}_text` | Post text content |
| `post_{1..15}_image_url` | Post image filename |
| `interaction_{1..5}_bio` | Bios of interacted users |
| `ground_truth` | Location ground truth |
| `ground_truth_age` | Age ground truth (optional) |
| `ground_truth_gender` | Gender ground truth (optional) |

### Batch Processing & Checkpoints

The pipeline processes users in configurable batches (default: 5). After each batch, results are saved to `checkpoints/results_checkpoint.json`. If the pipeline is interrupted, re-running the same command will automatically resume from the last checkpoint.

```bash
# Process 100 users, 5 at a time, with automatic checkpointing
demoinfer --csv data/large_dataset.csv --limit 100 --batch-size 5
```

## Project Structure

```
├── src/
│   └── demographic_inference/
│       ├── __init__.py          # Package init
│       ├── __main__.py          # python -m entry point
│       ├── cli.py               # CLI argument parsing
│       ├── config.py            # Configuration & API key resolution
│       ├── data_loader.py       # CSV ingestion & image loading
│       ├── llm_extractor.py     # Gemini API evidence extraction
│       ├── dst_engine.py        # DST math (combine, discount, fuse)
│       ├── pipeline.py          # Orchestrator with batch/checkpoint
│       └── metrics.py           # Accuracy & reporting
├── tests/
│   ├── test_dst_engine.py       # 29 DST math unit tests
│   └── test_data_loader.py      # 14 data loader tests
├── data/
│   └── sample_users.csv         # Sample test data
├── pyproject.toml               # Build config & dependencies
├── requirements.txt             # pip dependencies
├── .env.example                 # API key template
└── .gitignore
```

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Contributing — Git Quick Guide

### First-Time Setup (Do This Once)

```bash
# 1. Clone the repo to your machine
git clone https://github.com/MVB-glitch/demographic-inference.git
cd demographic-inference

# 2. Set up your identity (use your GitHub email)
git config user.name "Your Name"
git config user.email "your-email@example.com"

# 3. Install dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Daily Workflow — Making Changes

**Golden rule:** Never work directly on `main`. Always create a branch.

```bash
# 1. Make sure you're on main and up to date
git checkout main
git pull origin main

# 2. Create a new branch for your work
git checkout -b feature/improve-dst-engine
#                 ^^^^^^^^^^^^^^^^^^^^^^^^
#                 Name it after what you're doing

# 3. Make your changes (edit files, add features, fix bugs)
#    ... edit code ...

# 4. Check what you changed
git status                    # See which files changed
git diff                      # See the actual changes

# 5. Stage your changes
git add .                     # Stage ALL changes
# OR stage specific files:
git add src/demographic_inference/dst_engine.py

# 6. Commit with a clear message
git commit -m "Improve DST conflict tracking with cumulative metrics"

# 7. Push your branch to GitHub
git push origin feature/improve-dst-engine

# 8. Go to GitHub → Open a Pull Request → Get it reviewed → Merge
```

### Example: Adding a New Feature

```bash
# You want to add logging to the pipeline
git checkout main
git pull origin main
git checkout -b feature/add-logging

# Edit the files...
# src/demographic_inference/pipeline.py  ← add logging
# requirements.txt                       ← add new dependency if needed

# Test your changes
python -m pytest tests/ -v

# Commit and push
git add .
git commit -m "Add structured logging to pipeline"
git push origin feature/add-logging

# → Go to GitHub, create Pull Request, merge when ready
```

### Example: Fixing a Bug

```bash
git checkout main
git pull origin main
git checkout -b fix/mass-validation-edge-case

# Fix the bug...
# Add a test for the bug...

python -m pytest tests/ -v    # Make sure tests pass

git add .
git commit -m "Fix mass validation crash when Omega is missing"
git push origin fix/mass-validation-edge-case
```

### Useful Commands Cheat Sheet

| What you want to do | Command |
|----------------------|---------|
| See all branches | `git branch -a` |
| Switch to a branch | `git checkout branch-name` |
| See commit history | `git log --oneline -10` |
| Undo unstaged changes | `git checkout -- filename` |
| Undo last commit (keep files) | `git reset --soft HEAD~1` |
| Delete a local branch | `git branch -d branch-name` |
| Update your branch with latest main | `git checkout your-branch` then `git merge main` |
| See who changed a line | `git blame filename` |

### Branch Naming Convention

| Type | Format | Example |
|------|--------|---------|
| New feature | `feature/short-description` | `feature/add-logging` |
| Bug fix | `fix/short-description` | `fix/mass-validation` |
| Documentation | `docs/short-description` | `docs/update-readme` |
| Research/experiment | `research/short-description` | `research/calibration-study` |

### Pull Request Checklist

Before merging, make sure:
- [ ] Tests pass: `python -m pytest tests/ -v`
- [ ] Code runs without errors
- [ ] Commit messages are clear
- [ ] New features have corresponding tests
