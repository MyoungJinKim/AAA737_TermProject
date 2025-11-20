# /train/data/dataset.py

# /train/data/dataset.py

from __future__ import annotations
from typing import Dict, Optional

import logging
import io

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datasets import load_dataset, Audio
import soundfile as sf

from tokenizer import TextTokenizer
from .features import LogMelFeatureExtractor


class HuggingFaceSpeechDataset(Dataset):
    """
    HuggingFace audio dataset 래퍼.
    - log-mel feature 추출
    - 텍스트를 tokenizer로 인코딩
    - 길이/메타데이터 관리
    """

    def __init__(
        self,
        dataset_cfg: Dict,
        split: str,
        tokenizer: TextTokenizer,
        feature_extractor: LogMelFeatureExtractor,
        sample_rate: int,
        target_seconds: Optional[float] = None,
        pad_to_target: bool = False,
    ):
        # streaming 모드는 이 구현에서 지원하지 않음
        if dataset_cfg.get("streaming", False):
            raise ValueError("Streaming datasets are not supported.")

        self.dataset_cfg = dataset_cfg
        self.dataset_name = dataset_cfg["name"]
        self.split = split

        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate

        # 컬럼 이름 설정
        #  - audio_column 은 웬만하면 'audio'
        #  - text_column 은 기본값으로 'text' 를 쓰되, 실제 데이터셋에 없으면 뒤에서 fallback 시도
        self.audio_column = dataset_cfg.get("audio_column", "audio")
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

        # 오디오 컬럼을 지정한 sample_rate 로 캐스팅
        #ds = ds.cast_column(self.audio_column, Audio(sampling_rate=self.sample_rate))
        ds = ds.cast_column(self.audio_column, Audio(decode=False))

        # 실제 컬럼 목록
        col_names = list(ds.column_names)

        # ---- 텍스트 컬럼 존재 여부 확인 및 fallback ----
        if self.text_column not in col_names:
            # 1) MLS, Librispeech 계열은 보통 'transcript' 를 쓰므로 자동 fallback
            if self.text_column == "text" and "transcript" in col_names:
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
                # 2) 그 외에는 그냥 명시적으로 에러 + 사용 가능한 컬럼 리스트 출력
                raise ValueError(
                    f"Text column '{self.text_column}' not found in dataset '{self.dataset_name}' "
                    f"(split='{split}'). Available columns: {col_names}"
                )

        # 오디오 컬럼도 동일하게 체크
        if self.audio_column not in col_names:
            raise ValueError(
                f"Audio column '{self.audio_column}' not found in dataset '{self.dataset_name}' "
                f"(split='{split}'). Available columns: {col_names}"
            )

        self.dataset = ds

    def __len__(self) -> int:
        return self.dataset.num_rows

    @property
    def total_hours(self) -> float:
        """
        target_seconds 가 지정된 경우, 전체 데이터셋 길이를 대략적인 '시간(시간 단위)'으로 변환.
        (고정 길이 잘라 쓰는 CTC 학습에서 대략적인 정보용으로 사용)
        """
        if not self.target_num_frames:
            return 0.0
        fixed_sec = self.target_num_frames / self.sample_rate
        return len(self) * fixed_sec / 3600.0

    def _fix_duration(self, wav: torch.Tensor) -> torch.Tensor:
        """
        target_seconds 가 설정되어 있으면:
          - waveform 길이를 target_num_frames 로 잘라내거나(pad_to_target=False)
          - 필요 시 오른쪽 zero-padding(pad_to_target=True)
        """
        if self.target_num_frames is None:
            return wav

        n = wav.size(-1)

        # 길이가 너무 길면 자름
        if n > self.target_num_frames:
            return wav[..., : self.target_num_frames]

        # 길이가 짧으면 pad_to_target 옵션에 따라 오른쪽 zero-padding
        if n < self.target_num_frames and self.pad_to_target:
            pad = self.target_num_frames - n
            return F.pad(wav, (0, pad))

        return wav

    def __getitem__(self, idx: int) -> Dict:
        """
        개별 인덱스에서:
          - raw waveform 로드
          - 길이 조정
          - log-mel feature 추출
          - 텍스트 토큰 인코딩
          - 길이/메타데이터와 함께 반환
        """
        ex = self.dataset[idx]

        # ----- 오디오 로드 -----
        import torchaudio
        audio_dict = ex[self.audio_column]
        
        
        waveform, sr = torchaudio.load(io.BytesIO(audio_dict["bytes"]))  # file-like object 사용

        #byte_buf = io.BytesIO(audio_dict["bytes"])
        #data, sr = sf.read(byte_buf, dtype="float32")
        
        # datasets.Audio 로 캐스팅했으므로 'array' 키에 numpy 1D 또는 2D 가 들어있음
        wav = torch.tensor(waveform, dtype=torch.float32)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # [1, T]

        wav = self._fix_duration(wav)

        # ----- feature & token 추출 -----
        feats = self.feature_extractor(wav)  # [T_frames, F]

        # 텍스트 컬럼에서 문자열 읽어서 tokenizer로 인코딩
        text_value = ex[self.text_column]
        tokens = torch.tensor(
            self.tokenizer.encode(text_value),
            dtype=torch.long,
        )

        # duration(초 단위) 정보 계산
        duration = (
            self.target_seconds
            if self.target_seconds
            else wav.size(-1) / self.sample_rate
        )

        # utt_id 는 데이터셋에 id가 있으면 사용, 없으면 idx로 대체
        utt_id = ex.get("id", None)
        if utt_id is None:
            # MLS의 경우 original_path 등을 사용할 수도 있음
            utt_id = ex.get("original_path", idx)
        utt_id = str(utt_id)

        return {
            "features": feats,                # [T, F]
            "feature_length": feats.size(0),  # T
            "tokens": tokens,                 # [L]
            "token_length": tokens.size(0),   # L
            "utt_id": utt_id,
            "seconds": duration,
        }


# from __future__ import annotations
# from typing import Dict, Optional

# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset

# from datasets import load_dataset, Audio

# from tokenizer import TextTokenizer
# from .features import LogMelFeatureExtractor


# class HuggingFaceSpeechDataset(Dataset):
#     """
#     HF audio dataset wrapper producing:
#     - log-mel features
#     - encoded target tokens
#     - length metadata
#     """

#     def __init__(
#         self,
#         dataset_cfg: Dict,
#         split: str,
#         tokenizer: TextTokenizer,
#         feature_extractor: LogMelFeatureExtractor,
#         sample_rate: int,
#         target_seconds: Optional[float] = None,
#         pad_to_target: bool = False,
#     ):
#         if dataset_cfg.get("streaming", False):
#             raise ValueError("Streaming datasets are not supported.")

#         self.dataset_cfg = dataset_cfg
#         self.dataset_name = dataset_cfg["name"]
#         self.split = split

#         self.tokenizer = tokenizer
#         self.feature_extractor = feature_extractor
#         self.sample_rate = sample_rate

#         # columns
#         self.audio_column = dataset_cfg.get("audio_column", "audio")
#         self.text_column = dataset_cfg.get("text_column", "text")

#         # duration control
#         self.target_seconds = target_seconds
#         self.pad_to_target = pad_to_target
#         self.target_num_frames = (
#             int(round(target_seconds * sample_rate)) if target_seconds else None
#         )

#         # load dataset
#         load_kwargs = {
#             "split": split,
#             "cache_dir": dataset_cfg.get("cache_dir"),
#         }

#         config_name = dataset_cfg.get("config")
#         ds = (
#             load_dataset(self.dataset_name, config_name, **load_kwargs)
#             if config_name
#             else load_dataset(self.dataset_name, **load_kwargs)
#         )

#         # enforce audio sampling rate
#         ds = ds.cast_column(self.audio_column, Audio(sampling_rate=self.sample_rate))

#         if self.text_column not in ds.column_names:
#             raise ValueError(f"Text column '{self.text_column}' not found.")

#         if self.audio_column not in ds.column_names:
#             raise ValueError(f"Audio column '{self.audio_column}' not found.")

#         self.dataset = ds

#     def __len__(self):
#         return self.dataset.num_rows

#     @property
#     def total_hours(self) -> float:
#         if not self.target_num_frames:
#             return 0.0
#         fixed_sec = self.target_num_frames / self.sample_rate
#         return len(self) * fixed_sec / 3600.0

#     def _fix_duration(self, wav: torch.Tensor) -> torch.Tensor:
#         if self.target_num_frames is None:
#             return wav

#         n = wav.size(-1)

#         if n > self.target_num_frames:
#             return wav[..., : self.target_num_frames]

#         if n < self.target_num_frames and self.pad_to_target:
#             pad = self.target_num_frames - n
#             return F.pad(wav, (0, pad))

#         return wav

#     def __getitem__(self, idx: int) -> Dict:
#         ex = self.dataset[idx]

#         # load waveform
#         audio_dict = ex[self.audio_column]
#         wav = torch.tensor(audio_dict["array"], dtype=torch.float32)
#         if wav.dim() == 1:
#             wav = wav.unsqueeze(0)

#         wav = self._fix_duration(wav)

#         # features & tokens
#         feats = self.feature_extractor(wav)
#         tokens = torch.tensor(self.tokenizer.encode(ex[self.text_column]), dtype=torch.long)

#         duration = (
#             self.target_seconds
#             if self.target_seconds
#             else wav.size(-1) / self.sample_rate
#         )

#         return {
#             "features": feats,        # [T, F]
#             "feature_length": feats.size(0),
#             "tokens": tokens,
#             "token_length": tokens.size(0),
#             "utt_id": str(ex.get("id", idx)),
#             "seconds": duration,
#         }
