# /train/data/collate.py

from __future__ import annotations
from typing import Dict, List, Optional

import torch
from torch import Tensor


class SpeechDataCollator:
    """
    Encoder + 텍스트 학습용 Collator.

    - variable-length mel feature 시퀀스를 시간 축 기준 zero-padding
    - model.forward에서 바로 쓸 수 있도록
      {
        "input_features":      [B, T_max, F],
        "input_input_lengths": [B],
        "text":                List[str],
        "utt_id":              List[str],
      }
      형태의 딕셔너리를 반환한다.
    """

    def __init__(self, pad_to_multiple_of: Optional[int] = None):
        """
        Args:
            pad_to_multiple_of:
                - feature length(프레임 수)를 이 값의 배수까지 padding.
                - 예: 8로 설정 시, max time length를 8의 배수로 맞춰서
                  Conformer stride/subsampling과 깔끔하게 맞출 수 있음.
                - None이면 단순히 batch 내 max length까지만 padding.
        """
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict]) -> Dict[str, object]:
        """
        Args:
            batch:
                - HuggingFaceSpeechDataset.__getitem__ 의 반환값 리스트.

        Returns:
            batch_dict:
              {
                  "input_features":      Tensor[B, T_max, F],
                  "input_input_lengths": Tensor[B],  (각 샘플의 실제 길이)
                  "text":                List[str],
                  "utt_id":              List[str],
              }
        """
        assert len(batch) > 0, "Batch must contain at least one sample."

        # feature 차원(F)은 동일하다고 가정
        first_feat: Tensor = batch[0]["input_features"]
        feat_dim = first_feat.size(1)
        for s in batch:
            assert (
                s["input_features"].size(1) == feat_dim
            ), "All samples must share the same feature dimension (n_mels)."

        # 각 샘플의 시간 길이 T_i
        lengths = [int(s["feature_length"]) for s in batch]
        max_len = max(lengths)

        # 필요 시 pad_to_multiple_of 배수로 올림
        if self.pad_to_multiple_of is not None and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        B = len(batch)

        # dtype/device는 첫 feature 텐서를 기준으로 맞춤
        features = first_feat.new_zeros(B, max_len, feat_dim)  # [B, T_max, F]
        input_lengths = torch.as_tensor(lengths, dtype=torch.long)  # [B]

        utt_ids: List[str] = []
        texts: List[str] = []

        for i, s in enumerate(batch):
            L = int(s["feature_length"])
            feats_i = s["input_features"]          # [T_i, F]
            assert feats_i.size(0) == L, "feature_length와 실제 feature shape[0]가 다릅니다."

            # 앞쪽 L 프레임만 복사 (나머지는 0 패딩)
            features[i, :L] = feats_i

            utt_ids.append(s["utt_id"])
            texts.append(s["text"])

        # model.forward에서 바로 unpack 할 수 있도록 dict 형태로 반환
        return {
            "input_features": features,        # [B, T_max, F]
            "feature_length": input_lengths,  # [B]
            "text": texts,                     # 길이 B 리스트
            "utt_id": utt_ids,                # 디버깅/로깅용
        }
