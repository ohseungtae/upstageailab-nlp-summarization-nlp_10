import os
import pickle
import warnings
import sys
from pathlib import Path

# Set environment variables
os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")

# Python 3.12+ compatibility
if sys.version_info >= (3, 12):
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

import numpy as np
import pandas as pd

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="configs/",
    config_name="huggingface.yaml",
    version_base=None,
)
def merge_predictions(config: DictConfig) -> None:
    """Merge predictions from multiple devices/processes."""
    
    # Create Path objects for better path handling
    per_device_path = Path(config.per_device_save_path)
    connected_dir = Path(config.connected_dir)
    
    # Load logits from multiple devices
    logits_dir = per_device_path / "logits"
    if not logits_dir.exists():
        raise FileNotFoundError(f"Logits directory not found: {logits_dir}")
    
    logits = []
    for logit_file in logits_dir.glob("*.npy"):
        per_device_logit = np.load(logit_file)
        logits.append(per_device_logit)
    
    if not logits:
        raise ValueError(f"No logit files found in {logits_dir}")

    # Load generation dataframes from multiple devices
    generations_dir = per_device_path / "generations"
    if not generations_dir.exists():
        raise FileNotFoundError(f"Generations directory not found: {generations_dir}")
    
    generation_dfs = []
    for gen_file in generations_dir.glob("*.csv"):
        per_device_generation_df = pd.read_csv(gen_file)
        per_device_generation_df = per_device_generation_df.fillna("_")
        generation_dfs.append(per_device_generation_df)
    
    if not generation_dfs:
        raise ValueError(f"No generation files found in {generations_dir}")

    # Process logits
    all_logits = np.concatenate(logits, axis=0)
    unique_indices = np.unique(all_logits[:, :, -1])
    unique_indices = unique_indices.astype(int)
    unique_all_logits = all_logits[unique_indices]
    sorted_logits_with_indices = unique_all_logits[
        np.argsort(unique_all_logits[:, 0, -1])
    ]
    
    # Load submission template
    submission_file = connected_dir / "data" / f"{config.submission_file_name}.csv"
    if not submission_file.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_file}")
    
    generation_df = pd.read_csv(submission_file)
    sorted_logits = sorted_logits_with_indices[: len(generation_df), :, :-1]
    all_predictions = np.argmax(sorted_logits, axis=-1)
    
    # Save logits
    logits_output_dir = connected_dir / "logits"
    logits_output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(logits_output_dir / f"{config.logit_name}.pickle", "wb") as f:
        pickle.dump(sorted_logits, f)
    
    # Save predictions
    preds_output_dir = connected_dir / "preds"
    preds_output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(preds_output_dir / f"{config.pred_name}.pickle", "wb") as f:
        pickle.dump(all_predictions, f)

    # Process generations
    combined_generation_df = pd.concat(generation_dfs, ignore_index=True)
    sorted_generation_df = combined_generation_df.sort_values(by="index").reset_index(drop=True)
    all_generations = sorted_generation_df[config.target_column_name]
    
    # Validate generation length
    if len(all_generations) < len(generation_df):
        raise ValueError(
            f"Length of all_generations ({len(all_generations)}) is shorter than "
            f"length of predict data ({len(generation_df)})."
        )
    
    if len(all_generations) > len(generation_df):
        all_generations = all_generations[: len(generation_df)]
    
    # Update submission dataframe
    generation_df[config.target_column_name] = all_generations.values
    
    # Save final submission
    submissions_dir = connected_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    
    generation_df.to_csv(
        submissions_dir / f"{config.submission_name}.csv",
        index=False
    )
    
    print(f"Successfully merged predictions. Output saved to {submissions_dir / config.submission_name}.csv")


if __name__ == "__main__":
    merge_predictions()