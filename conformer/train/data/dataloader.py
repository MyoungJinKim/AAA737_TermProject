# /train/data/dataloader.py

from __future__ import annotations
from typing import Dict, Optional

from torch.utils.data import DataLoader, Dataset, Sampler

from tokenizer import TextTokenizer
from .features import LogMelFeatureExtractor
from .dataset import HuggingFaceSpeechDataset


def build_dataset(cfg: Dict, tokenizer: TextTokenizer, feature_extractor: LogMelFeatureExtractor, split: str):
    data_cfg = cfg["data"]
    hf_cfg = data_cfg.get("hf_dataset")

    if hf_cfg is None:
        raise ValueError("A Hugging Face dataset configuration is required.")

    return HuggingFaceSpeechDataset(
        hf_cfg,
        split=split,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        sample_rate=data_cfg["sample_rate"],
        target_seconds=data_cfg.get("max_audio_seconds"),
        pad_to_target=data_cfg.get("pad_to_max_seconds", False),
    )


def build_dataloader(
    dataset: Dataset,
    collate_fn,
    loader_cfg: Dict,
    shuffle: bool,
    sampler: Optional[Sampler] = None,
):
    kwargs = {
        "batch_size": loader_cfg["batch_size"],
        "num_workers": loader_cfg["num_workers"],
        "pin_memory": loader_cfg["pin_memory"],
        "persistent_workers": loader_cfg["persistent_workers"] and loader_cfg["num_workers"] > 0,
        "collate_fn": collate_fn,
        "drop_last": False,
    }

    if loader_cfg["num_workers"] > 0 and loader_cfg.get("prefetch_factor"):
        kwargs["prefetch_factor"] = loader_cfg["prefetch_factor"]

    if sampler is not None:
        # Sampler는 shuffle과 동시에 사용할 수 없으므로 우선 적용
        kwargs["sampler"] = sampler
        shuffle = False

    return DataLoader(dataset, shuffle=shuffle, **kwargs)
