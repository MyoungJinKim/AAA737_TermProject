import random
"""
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false  # 선택
export NCCL_P2P_DISABLE=1

torchrun \
  --standalone \
  --nproc_per_node=4 \
  train.py

"""

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils import *
from dist_utils import get_rank, init_distributed_mode
from model import load_model
from runner import Runner
import os
import yaml
from pathlib import Path
from data.dataloader import build_dataset, build_dataloader

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
CONFIG_PATH = Path("/data_x/aa007878/deep/myung/configs/config.yaml")


def load_config() -> dict:
    """
    YAML 설정 파일을 로드하고, device 필드를 기본값/실제 환경에 맞게 정리.
    """
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # device 필드 보정: cuda 설정인데 실제로 GPU가 없으면 cpu로 강제
    if cfg.get("device", "cuda") == "cuda" and not torch.cuda.is_available():
        cfg["device"] = "cpu"

    return cfg


def setup_seeds(base_seed: int):
    seed = base_seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    job_id = now()
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    config = load_config()
    device = torch.device(
            config.get(
                "device",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        )
    config["device"] = str(device)
    setup_seeds(config.get("seed", 1337))
        
    run_config = config.get("run", {})
    model_config = config.get("model", {})

    # initialize distributed training
    init_distributed_mode(run_config)
    setup_logger() # set after init_distributed_mode() to only log on master.

    # build model
    model = load_model(model_config)
    
    data_cfg = config["data"]
    hf_cfg = data_cfg["hf_dataset"]
    train_split = hf_cfg.get("train_split", "train")
    valid_split = hf_cfg.get("valid_split", "dev")

    # Dataset 생성
    train_dataset = build_dataset(config, split=train_split)
    valid_dataset = build_dataset(config, split=valid_split)

    # DataLoader 생성
    train_loader = build_dataloader(
        train_dataset,
        config["dataloader"],
        shuffle=config["dataloader"].get("shuffle", True),
    )
    valid_loader = build_dataloader(
        valid_dataset,
        config["dataloader"],
        shuffle=False,
    )

    # build runner
    runner = Runner(config, model, train_loader, valid_loader, job_id)

    # train
    runner.train()


if __name__ == "__main__":
    main()