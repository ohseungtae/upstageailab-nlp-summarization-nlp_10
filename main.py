import os
import warnings
import sys

# 2. torch 연산 정밀도 옵션 추가 (TensorCore 효율 + 메모리 절감)
import torch
torch.set_float32_matmul_precision('high')  # 또는 'high'도 실험 가능

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["HF_HOME"] = "/data/ephemeral/home/.cache/huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# 1. CUDA 환경변수(파편화 약화 옵션) 코드에서 직접 세팅 가능
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Filter warnings
warnings.filterwarnings("ignore")

# Check Python version compatibility
if sys.version_info >= (3, 12):
    # For Python 3.12+, set environment variable to use legacy behavior
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("CUDA cache is cleared.")
    
import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig

from src.pipelines.pipeline import train, test, predict, tune


@hydra.main(
    config_path="configs/",
    config_name="huggingface.yaml",
    version_base=None,  # Suppress hydra version warning
)
def main(config: DictConfig) -> None:
    """Main function to execute different modes of the pipeline."""
    
    if config.is_tuned == "tuned":
        tuned_params_path = Path(config.tuned_hparams_path)
        if not tuned_params_path.exists():
            raise FileNotFoundError(f"Tuned hyperparameters file not found: {config.tuned_hparams_path}")
        
        with open(tuned_params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        
        config = OmegaConf.merge(config, params)
    elif config.is_tuned == "untuned":
        pass
    else:
        raise ValueError(f"Invalid is_tuned argument: {config.is_tuned}")

    # Execute based on mode
    mode_functions = {
        "train": train,
        "test": test,
        "predict": predict,
        "tune": tune,
    }
    
    if config.mode not in mode_functions:
        raise ValueError(f"Invalid execution mode: {config.mode}. Available modes: {list(mode_functions.keys())}")
    
    return mode_functions[config.mode](config)


if __name__ == "__main__":
    main()