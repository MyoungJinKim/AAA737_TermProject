# /train/data/features.py

# /train/data/features.py

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class LogMelFeatureExtractor(nn.Module):
    """
    Log-Mel 스펙트로그램 + 간단한 시간축 평균/분산 정규화를 수행하는 모듈.

    입력:
        - waveform: [T] 또는 [C, T] 형태의 파형 텐서 (float, mono/stereo 등)
    출력:
        - mel: [Time, Mel] = [T_frames, n_mels]
          (torchaudio 기본 출력 [C, n_mels, T] → [T, n_mels] 로 변환)
    """

    def __init__(self, sample_rate: int, **kwargs):
        """
        Args:
            sample_rate: 오디오 샘플링 레이트 (예: 16000)
            kwargs:
                - n_mels: mel bin 개수 (기본 80)
                - n_fft, win_length, hop_length, f_min, f_max, mel_power, log_offset 등
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = kwargs.get("n_mels", 80)

        # torchaudio의 MelSpectrogram 변환 모듈
        # 출력 shape: [channel, n_mels, time]
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=kwargs.get("n_fft", 1024),
            win_length=kwargs.get("win_length", 400),
            hop_length=kwargs.get("hop_length", 160),
            f_min=kwargs.get("f_min", 0.0),
            f_max=kwargs.get("f_max", None),
            power=kwargs.get("mel_power", 2.0),
            n_mels=self.n_mels,
            norm=None,
            mel_scale="htk",
        )

        # log(0)을 피하기 위한 작은 offset
        self.log_offset = float(kwargs.get("log_offset", 1e-6))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [T] 또는 [C, T] 텐서
                - [T]: mono 1채널 파형
                - [C, T]: C채널 파형 (예: stereo)

        Returns:
            mel: [T_frames, n_mels] 텐서 (float32)
        """
        # torchaudio는 float32 파형을 기대하는 경우가 많으므로 강제 캐스팅
        waveform = waveform.to(dtype=torch.float32)

        # 1D 입력([T])이면 채널 차원을 추가 → [1, T]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]

        # 2D가 아니면 (예: [B, C, T] 같은 배치 단위 입력) 여기 설계와 맞지 않으므로 바로 에러
        assert waveform.dim() == 2, (
            f"Expected waveform shape [T] or [C, T], "
            f"but got {tuple(waveform.shape)}"
        )

        # 다채널 입력일 경우, 채널 평균을 내어 mono로 변환
        #   예: [2, T] (stereo) → [1, T] (mono)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # [1, T]

        # MelSpectrogram 추출
        # 출력: [channel=1, n_mels, time]
        mel = self.melspec(waveform)

        # log-mel 변환
        #  - mel 값이 0 또는 매우 작은 값이면 log에서 -inf가 나올 수 있으므로
        #    clamp_min_으로 하한을 log_offset 이상으로 올린 후 log_를 in-place로 적용
        mel = mel.clamp_min_(self.log_offset).log_()

        # 시간 축에 대한 평균/표준편차 정규화 (CMVN 비슷한 개념)
        #   현재 shape: [1, n_mels, time]
        #   dim=-1 기준: time 방향으로 평균/표준편차를 계산
        mel_mean = mel.mean(dim=-1, keepdim=True)
        mel_std = mel.std(dim=-1, keepdim=True)

        # (x - mean) / (std + eps)
        mel = (mel - mel_mean) / (mel_std + 1e-5)

        # 최종 출력 shape: [time, n_mels]
        #   [1, n_mels, time] → [1, time, n_mels] → [time, n_mels]
        mel = mel.transpose(1, 2).squeeze(0).contiguous()

        return mel

# from __future__ import annotations
# from typing import Optional, Dict

# import torch
# import torch.nn as nn
# import torchaudio


# class LogMelFeatureExtractor(nn.Module):
#     """
#     Produce log-mel spectrogram features with mean/variance normalization.
#     """

#     def __init__(self, sample_rate: int, **kwargs):
#         super().__init__()
#         self.sample_rate = sample_rate
#         self.n_mels = kwargs.get("n_mels", 80)

#         self.melspec = torchaudio.transforms.MelSpectrogram(
#             sample_rate=sample_rate,
#             n_fft=kwargs.get("n_fft", 1024),
#             win_length=kwargs.get("win_length", 400),
#             hop_length=kwargs.get("hop_length", 160),
#             f_min=kwargs.get("f_min", 0.0),
#             f_max=kwargs.get("f_max", None),
#             power=kwargs.get("mel_power", 2.0),
#             n_mels=self.n_mels,
#             norm=None,
#             mel_scale="htk",
#         )

#         self.log_offset = kwargs.get("log_offset", 1e-6)

#     def forward(self, waveform: torch.Tensor) -> torch.Tensor:
#         # mono
#         if waveform.dim() == 1:
#             waveform = waveform.unsqueeze(0)

#         # multi-channel → mono
#         if waveform.size(0) > 1:
#             waveform = waveform.mean(dim=0, keepdim=True)

#         mel = self.melspec(waveform)
#         mel = torch.log(torch.clamp(mel, min=self.log_offset))

#         # MVN normalization
#         mel = mel - mel.mean(dim=-1, keepdim=True)
#         mel = mel / (mel.std(dim=-1, keepdim=True) + 1e-5)

#         # output: [time, mel]
#         mel = mel.transpose(1, 2).squeeze(0).contiguous()
#         return mel
