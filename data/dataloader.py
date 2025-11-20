# /train/data/dataloader.py

from __future__ import annotations
from typing import Dict

from torch.utils.data import DataLoader, Dataset

from .features import LogMelFeatureExtractor
from .dataset import HuggingFaceSpeechDataset
from .collate import SpeechDataCollator


def build_dataset(cfg: Dict, split: str) -> HuggingFaceSpeechDataset:
    """
    cfg 예시 구조:
        cfg["data"] = {
            "sample_rate": 16000,
            "max_audio_seconds": 20.0,
            "pad_to_max_seconds": True,
            "hf_dataset": {...},
        }
        cfg["feature_extractor"] = {...}
    """
    data_cfg = cfg["data"]
    hf_cfg = data_cfg.get("hf_dataset")

    if hf_cfg is None:
        raise ValueError("A Hugging Face dataset configuration is required.")

    # Log-Mel feature extractor 생성
    feature_extractor = LogMelFeatureExtractor(
        sample_rate=data_cfg["sample_rate"],
        **cfg.get("feature_extractor", {}),
    )

    dataset = HuggingFaceSpeechDataset(
        dataset_cfg=hf_cfg,
        split=split,
        feature_extractor=feature_extractor,
        sample_rate=data_cfg["sample_rate"],
        target_seconds=data_cfg.get("max_audio_seconds"),
        pad_to_target=data_cfg.get("pad_to_max_seconds", False),
    )
    return dataset


def build_dataloader(dataset: Dataset, loader_cfg: Dict, shuffle: bool) -> DataLoader:
    """
    Encoder + text 학습용 DataLoader 빌더.
    Collate는 SpeechDataCollator를 사용해서
    'input_features', 'input_input_lengths', 'text' 를 포함한 배치를 반환한다.
    """
    collate_fn = SpeechDataCollator(
        pad_to_multiple_of=loader_cfg.get("pad_to_multiple_of")  # 없으면 None
    )

    kwargs = {
        "batch_size": loader_cfg["batch_size"],
        "num_workers": loader_cfg["num_workers"],
        "pin_memory": loader_cfg["pin_memory"],
        "persistent_workers": loader_cfg["persistent_workers"]
        and loader_cfg["num_workers"] > 0,
        "collate_fn": collate_fn,
        "drop_last": False,
    }

    if loader_cfg["num_workers"] > 0 and loader_cfg.get("prefetch_factor"):
        kwargs["prefetch_factor"] = loader_cfg["prefetch_factor"]

    return DataLoader(dataset, shuffle=shuffle, **kwargs)
