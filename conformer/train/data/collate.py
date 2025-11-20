# /train/data/collate.py

# /train/data/collate.py

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


class SpeechDataCollator:
    """
    CTC 학습을 위한 Collate 함수.
    - variable-length feature 시퀀스를 zero-padding해서 하나의 배치로 만듦
    - target 토큰 시퀀스는 CTC 규약에 따라 하나로 이어 붙여서(concatenate) 반환
    """

    def __init__(
        self,
        pad_to_multiple_of: Optional[int] = None,
        subsampling_factor: int = 1,
        min_subsample_len_multiplier: int = 1,
    ):
        """
        Args:
            pad_to_multiple_of:
                - feature length(프레임 길이)를 이 값의 배수까지 padding.
                - 예: 8이면 max_len을 8의 배수로 맞춰서 CNN/Conformer subsampling과 정렬을 쉽게 함.
            subsampling_factor:
                - encoder에서 시간 축으로 얼마나 줄어드는지 (예: stride 8이면 8).
                - _is_usable에서 "logit 길이 >= 토큰 길이" 체크에 사용.
            min_subsample_len_multiplier:
                - 최소 허용되는 "subsampled 길이"의 배수.
                - 극단적으로 짧은 발화를 버리기 위해 사용.
        """
        # subsampling_factor는 최소 1 이상이 되도록 방어 코드
        self.subsampling_factor = max(1, subsampling_factor)

        # subsampling 이후 길이 기준으로 너무 짧은 샘플을 필터링하기 위한 최소 프레임 수
        #   예) subsampling_factor=8, min_subsample_len_multiplier=2 이면
        #       최소 feature_length는 8 * 2 = 16 프레임
        self.min_subsample_frames = max(
            1, self.subsampling_factor * max(1, min_subsample_len_multiplier)
        )

        # max_len을 특정 배수로 맞춰 padding할지 여부
        self.pad_to_multiple_of = pad_to_multiple_of

    def _is_usable(self, s: Dict) -> bool:
        """
        너무 짧거나, CTC loss를 계산할 수 없는 샘플은 배치에서 제외한다.

        조건:
        1) token_length == 0  → 예측할 타겟이 없으므로 사용 불가
        2) feature_length < min_subsample_frames → subsampling 후 너무 짧음
        3) (feature_length // subsampling_factor) - 1 >= token_length
           를 만족해야 CTC 상에서 유효한 alignment가 가능하다고 가정
        """
        if s["token_length"] == 0:
            return False

        if s["feature_length"] < self.min_subsample_frames:
            return False

        # 대략적인 logit 길이 추정 (예: Conv/Conformer에서 subsampling_factor 만큼 줄어든다고 가정)
        approx_logits = max(1, (s["feature_length"] // self.subsampling_factor) - 1)

        # CTC에서는 logit 길이 >= 타겟 길이 여야 함
        return approx_logits >= s["token_length"]

    def __call__(
        self, batch: List[Dict]
    ) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, List[str]]]:
        """
        Args:
            batch:
                - dataset.__getitem__ 이 반환한 Dict들의 리스트.
                - 각 Dict는 최소한 다음 키를 가져야 한다:
                  "features": [T_i, F] tensor
                  "feature_length": int
                  "tokens": [L_i] tensor
                  "token_length": int
                  "utt_id": str

        Returns:
            (features, input_lengths, targets, target_lengths, utt_ids) 또는
            - batch 내 모든 샘플이 unusable일 경우 None (train loop에서 if batch is None: continue 로 처리)
        """
        # 1) unusable 샘플 필터링
        batch = [s for s in batch if self._is_usable(s)]
        if not batch:
            # 이 배치는 쓰지 않고 train loop에서 건너뜀
            return None

        # 2) feature 차원 정보 계산
        #    - 모든 샘플이 동일한 feature 차원을 갖는다고 가정
        first_feat: Tensor = batch[0]["features"]
        feat_dim = first_feat.size(1)

        # feature_dim이 다른 샘플이 끼어 있으면 조용히 깨지는 것보다, 즉시 에러를 내는 것이 안전
        for s in batch:
            assert (
                s["features"].size(1) == feat_dim
            ), "All samples in the batch must have the same feature dimension."

        # 3) padding할 최대 길이 계산 (시간 축 길이 T)
        max_len = max(s["feature_length"] for s in batch)

        # max_len을 pad_to_multiple_of 의 배수로 올림 (예: 123 → 128)
        if self.pad_to_multiple_of is not None and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        B = len(batch)

        # 4) 배치 텐서 초기화
        #    - dtype / device 는 첫 샘플을 그대로 따라감 (float32 CPU → float32 CPU 등)
        feats = first_feat.new_zeros(B, max_len, feat_dim)          # [B, T_max, F]
        in_lens = torch.zeros(B, dtype=torch.long)                  # [B]
        tgt_lens = torch.zeros(B, dtype=torch.long)                 # [B]

        targets: List[Tensor] = []
        utt_ids: List[str] = []

        # 5) 각 샘플을 배치 텐서에 채우기
        for i, s in enumerate(batch):
            L = s["feature_length"]

            # feature 복사: 맨 앞 L 프레임까지만 채우고, 나머지는 zero-padding 그대로 둠
            feats[i, :L] = s["features"]

            # 길이 정보 기록
            in_lens[i] = L
            tgt_lens[i] = s["token_length"]

            # target 토큰은 CTC loss에 맞게 1D로 이어 붙이기 위해 리스트에 모아둠
            targets.append(s["tokens"])
            utt_ids.append(s["utt_id"])

        # [sum(L_i)] 모양의 1D 텐서로 concat
        targets = torch.cat(targets, dim=0)

        # 최종 배치 반환
        #  - feats:       [B, T_max, F]
        #  - in_lens:     [B]
        #  - targets:     [sum(L_i)]
        #  - tgt_lens:    [B]
        #  - utt_ids:     길이 B인 문자열 리스트
        return feats, in_lens, targets, tgt_lens, utt_ids


# from __future__ import annotations
# from typing import Dict, List, Optional

# import torch


# class SpeechDataCollator:
#     """
#     Collate function for CTC-style training.
#     Pads feature sequences + concatenates tokens.
#     """

#     def __init__(
#         self,
#         pad_to_multiple_of: Optional[int] = None,
#         subsampling_factor: int = 1,
#         min_subsample_len_multiplier: int = 1,
#     ):
#         self.pad_to_multiple_of = pad_to_multiple_of
#         self.subsampling_factor = max(1, subsampling_factor)
#         self.min_subsample_frames = max(
#             1, self.subsampling_factor * max(1, min_subsample_len_multiplier)
#         )

#     def _is_usable(self, s: Dict) -> bool:
#         if s["token_length"] == 0:
#             return False
#         if s["feature_length"] < self.min_subsample_frames:
#             return False
#         approx_logits = max(1, (s["feature_length"] // self.subsampling_factor) - 1)
#         return approx_logits >= s["token_length"]

#     def __call__(self, batch: List[Dict]):
#         batch = [s for s in batch if self._is_usable(s)]
#         if not batch:
#             return None

#         feat_dim = batch[0]["features"].size(1)
#         max_len = max(s["feature_length"] for s in batch)

#         if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of != 0:
#             max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

#         B = len(batch)
#         feats = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
#         in_lens = torch.zeros(B, dtype=torch.long)
#         tgt_lens = torch.zeros(B, dtype=torch.long)

#         targets = []
#         utt_ids = []

#         for i, s in enumerate(batch):
#             L = s["feature_length"]
#             feats[i, :L] = s["features"]
#             in_lens[i] = L
#             tgt_lens[i] = s["token_length"]
#             targets.append(s["tokens"])
#             utt_ids.append(s["utt_id"])

#         targets = torch.cat(targets)

#         return feats, in_lens, targets, tgt_lens, utt_ids
