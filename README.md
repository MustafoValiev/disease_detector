# Cucumber Disease Detector

This repository contains a PyTorch-based model training script for cucumber disease classification.

## Overview

- `train.py`: Main training script.
- `train.ipynb`: Jupyter notebook version of the training workflow.
- `confusion_matrix.png`: Saved confusion matrix from evaluation.
- `training_curves_optimized.png`: Saved training/test curves.
- `requirements.txt`: Python dependencies.

## Setup

1. Create a Python virtual environment:

   ```bash
   python -m venv venv
   ```

2. Activate the environment:

   - Windows PowerShell:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - Windows CMD:
     ```cmd
     .\venv\Scripts\activate.bat
     ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

Place your dataset under the `data/` directory in the following structure:

```
data/
  class_1/
    image1.jpg
    image2.jpg
  class_2/
    image1.jpg
    image2.jpg
  ...
```

The script assumes `NUM_CLASSES = 8` by default. If your dataset contains a different number of classes, update `NUM_CLASSES` in `train.py`.

## Run training

```bash
python train.py
```

The script will:

- load images from `./data`
- perform train/test split
- train the CNN model
- save `training_curves_optimized.png`
- save `confusion_matrix.png`

## Notes

- If a GPU is available, the script will use it automatically.
- `data/` is ignored by `.gitignore` to avoid committing large datasets.
- Model weights are not saved by default in `train.py`; add a `torch.save(...)` call if you want to persist checkpoints.

## GitHub

This folder is configured to use the remote repository:

`https://github.com/MustafoValiev/disease_detector.git`

To push changes:

```bash
git add .
git commit -m "Add project README and gitignore"
git push origin main
```
