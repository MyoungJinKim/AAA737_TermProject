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
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = kwargs.get("n_mels", 80)

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
        waveform = waveform.to(dtype=torch.float32)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]

        assert waveform.dim() == 2, (
            f"Expected waveform shape [T] or [C, T], "
            f"but got {tuple(waveform.shape)}"
        )

        # 다채널 → mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # [1, T]

        # [1, n_mels, time]
        mel = self.melspec(waveform)

        # log-mel
        mel = mel.clamp_min_(self.log_offset).log_()

        # 시간 축 기준 CMVN
        mel_mean = mel.mean(dim=-1, keepdim=True)
        mel_std = mel.std(dim=-1, keepdim=True)
        mel = (mel - mel_mean) / (mel_std + 1e-5)

        # [1, n_mels, T] → [1, T, n_mels] → [T, n_mels]
        mel = mel.transpose(1, 2).squeeze(0).contiguous()
        return mel
