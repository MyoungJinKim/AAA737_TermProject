import argparse
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from evaluate import load as load_metric

from data.dataloader import build_dataset, build_dataloader
from model import load_model
from utils import prepare_sample, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute WER using HuggingFace evaluate.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="학습/데이터 설정 YAML 경로")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None, #여기 !!!
        help="stage2 checkpoint(.pth) 경로",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("device", "cuda") == "cuda" and not torch.cuda.is_available():
        cfg["device"] = "cpu"

    return cfg


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {unexpected}")


def main():
    args = parse_args()
    setup_logger()

    cfg = load_config(args.config)
    device = torch.device(
        cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    cfg["device"] = str(device)

    model = load_model(cfg["model"])
    model.to(device)
    model.device = device
    model.eval()

    if args.ckpt:
        load_checkpoint(model, args.ckpt)
    else:
        print("[WARN] No checkpoint provided; evaluating with initial weights.")

    dataset = build_dataset(cfg, split="test")
    dataloader = build_dataloader(dataset, cfg.get("dataloader", {}), shuffle=False)

    generate_cfg = cfg.get("generate", cfg.get("run", {}))
    wer_metric = load_metric("wer")
    use_cuda = device.type == "cuda"

    evaluated = 0
    skipped_zero_ref = 0
    references_all: List[str] = []
    predictions_all: List[str] = []

    with torch.inference_mode():
        for batch in dataloader:
            references = batch["text"]

            samples = prepare_sample(batch, cuda_enabled=use_cuda)
            predictions = model.generate(samples, generate_cfg)

            for ref, hyp in zip(references, predictions):
                if not ref or not ref.strip():
                    skipped_zero_ref += 1
                    continue

                references_all.append(ref)
                predictions_all.append(hyp)
                evaluated += 1

    if not references_all:
        print("평가할 참조 텍스트가 없습니다. (빈 reference만 존재)")
        return

    wer = wer_metric.compute(references=references_all, predictions=predictions_all)
    print(f"Evaluated samples : {evaluated}")
    print(f"Word Error Rate   : {wer * 100:.2f}%")
    if skipped_zero_ref:
        print(f"Skipped samples with empty reference: {skipped_zero_ref}")


if __name__ == "__main__":
    main()