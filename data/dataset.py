# /train/data/dataset.py

from __future__ import annotations
from typing import Dict, Optional

import logging

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset, Audio

from .features import LogMelFeatureExtractor


class HuggingFaceSpeechDataset(Dataset):
    """
    HuggingFace audio dataset 래퍼.
    - raw waveform을 HF에서 로드
    - 길이 조정(잘라내기/패딩, 옵션)
    - Log-Mel feature 추출
    - 텍스트는 raw string으로 그대로 보존

    CTC/토큰 관련 로직은 전부 제거하고,
    Conformer encoder + text 학습(예: LLM과 연결)용으로 단순화한 버전.
    """

    def __init__(
        self,
        dataset_cfg: Dict,
        split: str,
        feature_extractor: LogMelFeatureExtractor,
        sample_rate: int,
        target_seconds: Optional[float] = None,
        pad_to_target: bool = False,
    ):
        if dataset_cfg.get("streaming", False):
            raise ValueError("Streaming datasets are not supported.")

        self.dataset_cfg = dataset_cfg
        self.dataset_name = dataset_cfg["name"]
        self.split = split

        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate

        # 컬럼 이름 설정
        self.audio_column = dataset_cfg.get("audio_column", "audio")
        # 기본 텍스트 컬럼 후보: 'text' → 없으면 'transcript' 로 fallback
        self.text_column = dataset_cfg.get("text_column", "transcript")

        # duration (길이 고정/패딩) 설정
        self.target_seconds = target_seconds
        self.pad_to_target = pad_to_target
        self.target_num_frames = (
            int(round(target_seconds * sample_rate)) if target_seconds else None
        )

        # HF dataset 로드
        load_kwargs = {
            "split": split,
            "cache_dir": dataset_cfg.get("cache_dir"),
        }
        config_name = dataset_cfg.get("config")
        ds = (
            load_dataset(self.dataset_name, config_name, **load_kwargs)
            if config_name
            else load_dataset(self.dataset_name, **load_kwargs)
        )

        # 버전 때문에 어쩔 수 없음 
        ds = ds.cast_column(self.audio_column, Audio(decode=False))

        col_names = list(ds.column_names)

        # ---- 텍스트 컬럼 존재 여부 확인 + fallback ----
        if self.text_column not in col_names:
            # text가 없고 transcript가 있으면 transcript로 자동 변경
            if "transcript" in col_names:
                logging.warning(
                    "Text column '%s' not found in dataset '%s' (split=%s). "
                    "Falling back to 'transcript'. Available columns: %s",
                    self.text_column,
                    self.dataset_name,
                    split,
                    col_names,
                )
                self.text_column = "transcript"
            else:
                raise ValueError(
                    f"Text column '{self.text_column}' not found in dataset "
                    f"'{self.dataset_name}' (split='{split}'). "
                    f"Available columns: {col_names}"
                )

        # 오디오 컬럼 체크
        if self.audio_column not in col_names:
            raise ValueError(
                f"Audio column '{self.audio_column}' not found in dataset "
                f"'{self.dataset_name}' (split='{split}'). "
                f"Available columns: {col_names}"
            )

        self.dataset = ds

    def __len__(self) -> int:
        return self.dataset.num_rows

    @property
    def total_hours(self) -> float:
        """
        target_seconds 를 사용하는 경우,
        '매 샘플이 target_seconds 만큼의 오디오를 가진다'고 가정하고
        전체 길이를 시간 단위로 대략 계산.
        """
        if not self.target_num_frames:
            return 0.0
        fixed_sec = self.target_num_frames / self.sample_rate
        return len(self) * fixed_sec / 3600.0

    def _fix_duration(self, wav: torch.Tensor) -> torch.Tensor:
        """
        target_seconds 가 설정되어 있으면:
          - 길이가 target_num_frames 보다 길면 잘라내고
          - pad_to_target=True 이면 짧은 경우 오른쪽을 zero-padding
        """
        if self.target_num_frames is None:
            return wav

        n = wav.size(-1)

        # 길이가 너무 길면 자르기
        if n > self.target_num_frames:
            return wav[..., : self.target_num_frames]

        # 길이가 짧으면 pad_to_target 옵션에 따라 오른쪽 zero-padding
        if n < self.target_num_frames and self.pad_to_target:
            pad = self.target_num_frames - n
            return F.pad(wav, (0, pad))

        return wav

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            {
                "input_features": [T_i, F]  (Log-Mel feature),
                "feature_length": T_i,      (길이, collator에서 텐서로 묶음)
                "text": str,                (raw transcript)
                "utt_id": str,              (샘플 ID)
            }
        """
        ex = self.dataset[idx]

        # ----- 오디오 로드 -----
        audio_dict = ex[self.audio_column]  # {"bytes": np.ndarray, ...}
        wav = torch.tensor(audio_dict["bytes"], dtype=torch.float32)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # [1, T]

        wav = self._fix_duration(wav)

        # ----- feature 추출 -----
        feats = self.feature_extractor(wav)  # [T_i, F]
        feat_len = feats.size(0)

        # 텍스트는 raw string으로 가져옴
        text_value = ex[self.text_column]

        # utt_id: 있으면 사용, 없으면 idx로 대체
        utt_id = ex.get("id", None)
        if utt_id is None:
            utt_id = ex.get("original_path", idx)
        utt_id = str(utt_id)

        return {
            "input_features": feats,  # [T_i, F]
            "feature_length": feat_len,
            "text": text_value,
            "utt_id": utt_id,
        }
