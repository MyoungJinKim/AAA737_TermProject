# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from .activation import Swish, GLU
from .modules import Transpose


class DepthwiseConv1d(nn.Module):
    """
    Depthwise Convolution (1D).
    groups=in_channels 로 설정하면, 각 채널마다 독립적으로 convolution이 적용됨.
    → 채널 간 정보 혼합 없이 channel-wise filtering 을 수행하는 구조.

    딥러닝 문헌에서는
    out_channels = K * in_channels 일 때 depthwise convolution으로 불림.
    (K = depth multiplier)
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()

        # Depthwise Conv 조건: out_channels = k * in_channels
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"

        # Conv1d에서 groups=in_channels 이면, 채널별로 따로 convolution 수행
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,      # 핵심: depthwise convolution
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # 입력: (B, C, T) → 출력: (B, C*K, T)
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    """
    Pointwise Convolution (kernel_size=1).
    Conv1d with kernel_size=1은 channel mixing만 수행하며,
    시간축(T)은 그대로 유지됨.
    주로 차원 조정 혹은 채널 간 정보 결합에 사용됨.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()

        # 1x1 convolution → 채널 차원만 변경
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,          # 핵심: pointwise conv
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # 입력: (B, C, T) → 출력: (B, C', T)
        return self.conv(inputs)


class ConformerConvModule(nn.Module):
    """
    Conformer의 Convolution Module.
    Conformer 블록의 핵심 구성요소 중 하나로, 다음 순서로 진행됨:

    1) LayerNorm
    2) PointwiseConv + GLU
          - GLU는 입력을 gate와 value로 나누어 (value * sigmoid(gate))
            형태로 gating 적용
    3) DepthwiseConv
          - 'SAME' padding 효과를 위해 kernel_size는 홀수여야 한다
    4) BatchNorm
    5) Swish Activation
    6) PointwiseConv (출력 차원 복원)
    7) Dropout

    최종적으로 (B, T, C) → (B, T, C)
    """
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConformerConvModule, self).__init__()

        # SAME padding을 위해 kernel_size는 홀수여야 함
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"

        # GLU 사용 시 expansion_factor=2 필요 (GLU 내부에서 두 채널로 split)
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),              # (B, T, C) 기준으로 layer norm
            Transpose(shape=(1, 2)),                # (B, C, T) 형태로 변환 (Conv1d 요구 형식)

            # 1) Pointwise conv로 channel 확장 → GLU에서 절반은 gate로 사용
            PointwiseConv1d(in_channels, in_channels * expansion_factor),

            # 2) Gated Linear Unit
            GLU(dim=1),                             # dim=1 → 채널 차원으로 split

            # 3) Depthwise Conv (local context learning)
            DepthwiseConv1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,     # SAME padding 유지
            ),

            nn.BatchNorm1d(in_channels),            # conv 결과 stabilize
            Swish(),                               # non-linear activation

            # 4) PointwiseConv로 다시 채널 복원
            PointwiseConv1d(in_channels, in_channels),

            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # 입력: (B, T, C) → transpose → (B, C, T) → sequential → (B, C, T)
        # 다시 transpose → (B, T, C)
        return self.sequential(inputs).transpose(1, 2)


class Conv2dSubampling(nn.Module):
    """
    2D Convolution 기반 Subsampling.
    일반적으로 Transformer Encoder 입력 길이(T)를 1/8로 줄이기 위해 사용.

    두 번의 Conv2d(stride=2) → 길이 절반 → 다시 절반 = 원래의 1/8.
    ASR Conformer 구조에서 필수적으로 등장하는 모듈.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv2dSubampling, self).__init__()

        # 인풋 형태 (B, T, F)를 2D로 만들기 위해 unsqueeze(1) → (B, 1, T, F)
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=8),
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:

        # inputs: (B, T, F)
        outputs = self.sequential(inputs.unsqueeze(1))   # → (B, C, T/8, F/8)

        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

        # 2D conv 출력 → (B, T/8, C*F')
        outputs = outputs.permute(0, 2, 1, 3)            # (B, T', C, F')
        outputs = outputs.contiguous().view(
            batch_size,
            subsampled_lengths,
            channels * sumsampled_dim,
        )

        # 길이도 동일하게 1/8로 줄어듦
        output_lengths = input_lengths >> 3   # right shift = integer division by 8
        output_lengths -= 1                  # conv kernel 영향 보정

        return outputs, output_lengths


# class DepthwiseConv1d(nn.Module):
#     """
#     When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
#     this operation is termed in literature as depthwise convolution.

#     Args:
#         in_channels (int): Number of channels in the input
#         out_channels (int): Number of channels produced by the convolution
#         kernel_size (int or tuple): Size of the convolving kernel
#         stride (int, optional): Stride of the convolution. Default: 1
#         padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
#         bias (bool, optional): If True, adds a learnable bias to the output. Default: True

#     Inputs: inputs
#         - **inputs** (batch, in_channels, time): Tensor containing input vector

#     Returns: outputs
#         - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
#     """
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             kernel_size: int,
#             stride: int = 1,
#             padding: int = 0,
#             bias: bool = False,
#     ) -> None:
#         super(DepthwiseConv1d, self).__init__()
#         assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
#         self.conv = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             groups=in_channels,
#             stride=stride,
#             padding=padding,
#             bias=bias,
#         )

#     def forward(self, inputs: Tensor) -> Tensor:
#         return self.conv(inputs)


# class PointwiseConv1d(nn.Module):
#     """
#     When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
#     This operation often used to match dimensions.

#     Args:
#         in_channels (int): Number of channels in the input
#         out_channels (int): Number of channels produced by the convolution
#         stride (int, optional): Stride of the convolution. Default: 1
#         padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
#         bias (bool, optional): If True, adds a learnable bias to the output. Default: True

#     Inputs: inputs
#         - **inputs** (batch, in_channels, time): Tensor containing input vector

#     Returns: outputs
#         - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
#     """
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             stride: int = 1,
#             padding: int = 0,
#             bias: bool = True,
#     ) -> None:
#         super(PointwiseConv1d, self).__init__()
#         self.conv = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=1,
#             stride=stride,
#             padding=padding,
#             bias=bias,
#         )

#     def forward(self, inputs: Tensor) -> Tensor:
#         return self.conv(inputs)


# class ConformerConvModule(nn.Module):
#     """
#     Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).
#     This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
#     to aid training deep models.

#     Args:
#         in_channels (int): Number of channels in the input
#         kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
#         dropout_p (float, optional): probability of dropout

#     Inputs: inputs
#         inputs (batch, time, dim): Tensor contains input sequences

#     Outputs: outputs
#         outputs (batch, time, dim): Tensor produces by conformer convolution module.
#     """
#     def __init__(
#             self,
#             in_channels: int,
#             kernel_size: int = 31,
#             expansion_factor: int = 2,
#             dropout_p: float = 0.1,
#     ) -> None:
#         super(ConformerConvModule, self).__init__()
#         assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
#         assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

#         self.sequential = nn.Sequential(
#             nn.LayerNorm(in_channels),
#             Transpose(shape=(1, 2)),
#             PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
#             GLU(dim=1),
#             DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
#             nn.BatchNorm1d(in_channels),
#             Swish(),
#             PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
#             nn.Dropout(p=dropout_p),
#         )

#     def forward(self, inputs: Tensor) -> Tensor:
#         return self.sequential(inputs).transpose(1, 2)


# class Conv2dSubampling(nn.Module):
#     """
#     Convolutional 2D subsampling (to 1/4 length)

#     Args:
#         in_channels (int): Number of channels in the input image
#         out_channels (int): Number of channels produced by the convolution

#     Inputs: inputs
#         - **inputs** (batch, time, dim): Tensor containing sequence of inputs

#     Returns: outputs, output_lengths
#         - **outputs** (batch, time, dim): Tensor produced by the convolution
#         - **output_lengths** (batch): list of sequence output lengths
#     """
#     def __init__(self, in_channels: int, out_channels: int) -> None:
#         super(Conv2dSubampling, self).__init__()
#         self.sequential = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
#             nn.ReLU(),
#         )

#     def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
#         outputs = self.sequential(inputs.unsqueeze(1))
#         batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

#         outputs = outputs.permute(0, 2, 1, 3)
#         outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)

#         output_lengths = input_lengths >> 2
#         output_lengths -= 1

#         return outputs, output_lengths
