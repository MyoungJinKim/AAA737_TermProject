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

from .feed_forward import FeedForwardModule
from .attention import MultiHeadedSelfAttentionModule
from .convolution import (
    ConformerConvModule,
    Conv2dSubampling,
)
from .modules import (
    ResidualConnectionModule,
    Linear,
)

class ConformerBlock(nn.Module):
    """
    Conformer 블록은 Macaron-Net 구조를 따르는 블록으로,
    FFN(Feed Forward Network) - MHSA(Multi-Headed Self-Attention) - ConvModule - FFN
    을 residual로 쌓은 구조이다.

    특히 Feed Forward 모듈을 앞/뒤에 반씩 나누어 두 번 적용하는데,
    이를 Macaron 스타일이라고 하며, 원래 FFN의 출력에 1.0을 곱하던 residual을
    앞/뒤 각각 0.5씩 나누어 사용하는 개념(half-step residual)이다.
    """
    def __init__(
            self,
            encoder_dim: int = 512,                  # Conformer 블록의 hidden dimension
            num_attention_heads: int = 8,           # MHSA head 개수
            feed_forward_expansion_factor: int = 4, # FFN 내부에서 hidden dimension 확장 비율
            conv_expansion_factor: int = 2,         # ConvModule에서 GLU 전 확장 비율
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = False,        # Macaron 스타일(0.5 residual) 사용 여부
    ):
        super(ConformerBlock, self).__init__()

        # Macaron-Net: FFN을 두 번 쓰되, 각 FFN residual 스케일을 0.5로 줄이는 방식
        # if half_step_residual:
        #     self.feed_forward_residual_factor = 0.5
        # else:
        #     self.feed_forward_residual_factor = 1

        self.feed_forward_residual_factor = 1            

        # Conformer 블록은 다음과 같은 순서를 가짐:
        # 1) (Scaled) FeedForward
        # 2) Multi-Headed Self-Attention
        # 3) Convolution Module
        # 4) (Scaled) FeedForward
        # 5) LayerNorm
        self.sequential = nn.Sequential(
            # 1번째 FFN (앞단 FFN) + Residual
            # ResidualConnectionModule(
            #     module=FeedForwardModule(
            #         encoder_dim=encoder_dim,
            #         expansion_factor=feed_forward_expansion_factor,
            #         dropout_p=feed_forward_dropout_p,
            #     ),
            #     module_factor=self.feed_forward_residual_factor,  # residual scaling (0.5 or 1.0)
            # ),

            # MHSA + Residual
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),

            # Conformer Conv Module + Residual
            ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),

            # 2번째 FFN (뒷단 FFN) + Residual
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,  # 역시 half-step residual
            ),

            # 마지막 LayerNorm (Pre-Norm 구조에서 block 끝 단 정규화)
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs: (B, T, D) → 동일 shape 유지
        return self.sequential(inputs)


class ConformerEncoder(nn.Module):
    """
    Conformer Encoder 전체 구조.

    1) Conv2dSubsampling 으로 입력 시퀀스 길이(T)를 1/4로 줄이고,
       feature dimension(F)를 conv2d를 통해 embedding dimension으로 맵핑하기 용이한 형태로 만듦.
    2) Linear projection으로 Conv2d 출력 채널*feature를 encoder_dim으로 사상.
    3) 여러 개의 ConformerBlock 을 stack 하여 인코더를 구성.

    입력:  (B, T, input_dim)   - 예: input_dim=80 (Mel filter bank 차원)
    출력:  (B, T', encoder_dim)
    """
    def __init__(
            self,
            input_dim: int = 80,                     # 원본 feature dimension (예: 80-dim fbank)
            encoder_dim: int = 512,                 # 인코더 hidden dim
            num_layers: int = 17,                   # ConformerBlock 개수
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ):
        super(ConformerEncoder, self).__init__()

        # 2D Conv 기반 서브샘플링 모듈
        # 입력: (B, T, F) → unsqueeze(1) → (B, 1, T, F)
        # Conv2dSubampling(in_channels=1, out_channels=encoder_dim) 이므로
        # 출력 채널: encoder_dim, 길이: T/8, feature: F' (대략 F/8)
        self.conv_subsample = Conv2dSubampling(in_channels=1, out_channels=encoder_dim)

        # Conv2dSubampling 이후 출력 shape:
        # outputs: (B, T', C*F') 로 flatten 되어 나옴
        # 여기서 C=encoder_dim, F'는 conv 결과 feature 길이
        #
        # Conv2dSubampling에서 feature 축(F)에 대해 두 번 (kernel=3,stride=2) conv를 적용하므로,
        # 대략 F' = (((input_dim - 1) // 2 - 1) // 2)
        #
        # 따라서 최종 feature dimension = encoder_dim * F'
        # → 이를 Linear로 encoder_dim으로 축소(projection)
        self.input_projection = nn.Sequential(
            Linear(
                encoder_dim *  (((input_dim - 3) // 8) + 1),  # Conv2d 후 flatten된 feature 차원
                encoder_dim,
            ),
            nn.Dropout(p=input_dropout_p),
        )

        # ConformerBlock N개 스택
        self.layers = nn.ModuleList([
            ConformerBlock(
                encoder_dim=encoder_dim,
                num_attention_heads=num_attention_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                feed_forward_dropout_p=feed_forward_dropout_p,
                attention_dropout_p=attention_dropout_p,
                conv_dropout_p=conv_dropout_p,
                conv_kernel_size=conv_kernel_size,
                half_step_residual=half_step_residual,
            )
            for _ in range(num_layers)
        ])

    def count_parameters(self) -> int:
        """ Encoder 전체 파라미터 수 카운트 """
        return sum([p.numel() for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """
        Encoder 내 Dropout 확률을 일괄적으로 변경하고 싶을 때 사용.
        (현재 구현은 top-level child 중 nn.Dropout에만 적용됨)
        """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encoder 전방향 패스.

        Args:
            inputs:
                - shape: (B, T, D_in), 보통 fbank 등의 음성 feature 시퀀스
            input_lengths:
                - shape: (B,), 각 샘플의 실제 길이(T)

        Returns:
            outputs:
                - shape: (B, T', encoder_dim), T' ≈ T/8
            output_lengths:
                - shape: (B,), conv subsampling 이후의 시퀀스 길이
        """
        # 1) Conv2d 기반 서브샘플링
        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        # outputs: (B, T', encoder_dim * F'),  output_lengths: 대략 T/8

        # 2) Linear projection으로 encoder_dim으로 압축
        outputs = self.input_projection(outputs)   # (B, T', encoder_dim)

        # 3) 여러 층의 ConformerBlock 통과
        for layer in self.layers:
            outputs = layer(outputs)               # 각 layer: (B, T', encoder_dim) 유지

        return outputs, output_lengths



# class ConformerBlock(nn.Module):
#     """
#     Conformer block contains two Feed Forward modules sandwiching the Multi-Headed Self-Attention module
#     and the Convolution module. This sandwich structure is inspired by Macaron-Net, which proposes replacing
#     the original feed-forward layer in the Transformer block into two half-step feed-forward layers,
#     one before the attention layer and one after.

#     Args:
#         encoder_dim (int, optional): Dimension of conformer encoder
#         num_attention_heads (int, optional): Number of attention heads
#         feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
#         conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
#         feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
#         attention_dropout_p (float, optional): Probability of attention module dropout
#         conv_dropout_p (float, optional): Probability of conformer convolution module dropout
#         conv_kernel_size (int or tuple, optional): Size of the convolving kernel
#         half_step_residual (bool): Flag indication whether to use half step residual or not

#     Inputs: inputs
#         - **inputs** (batch, time, dim): Tensor containing input vector

#     Returns: outputs
#         - **outputs** (batch, time, dim): Tensor produces by conformer block.
#     """
#     def __init__(
#             self,
#             encoder_dim: int = 512,
#             num_attention_heads: int = 8,
#             feed_forward_expansion_factor: int = 4,
#             conv_expansion_factor: int = 2,
#             feed_forward_dropout_p: float = 0.1,
#             attention_dropout_p: float = 0.1,
#             conv_dropout_p: float = 0.1,
#             conv_kernel_size: int = 31,
#             half_step_residual: bool = True,
#     ):
#         super(ConformerBlock, self).__init__()
#         if half_step_residual:
#             self.feed_forward_residual_factor = 0.5
#         else:
#             self.feed_forward_residual_factor = 1

#         self.sequential = nn.Sequential(
#             ResidualConnectionModule(
#                 module=FeedForwardModule(
#                     encoder_dim=encoder_dim,
#                     expansion_factor=feed_forward_expansion_factor,
#                     dropout_p=feed_forward_dropout_p,
#                 ),
#                 module_factor=self.feed_forward_residual_factor,
#             ),
#             ResidualConnectionModule(
#                 module=MultiHeadedSelfAttentionModule(
#                     d_model=encoder_dim,
#                     num_heads=num_attention_heads,
#                     dropout_p=attention_dropout_p,
#                 ),
#             ),
#             ResidualConnectionModule(
#                 module=ConformerConvModule(
#                     in_channels=encoder_dim,
#                     kernel_size=conv_kernel_size,
#                     expansion_factor=conv_expansion_factor,
#                     dropout_p=conv_dropout_p,
#                 ),
#             ),
#             ResidualConnectionModule(
#                 module=FeedForwardModule(
#                     encoder_dim=encoder_dim,
#                     expansion_factor=feed_forward_expansion_factor,
#                     dropout_p=feed_forward_dropout_p,
#                 ),
#                 module_factor=self.feed_forward_residual_factor,
#             ),
#             nn.LayerNorm(encoder_dim),
#         )

#     def forward(self, inputs: Tensor) -> Tensor:
#         return self.sequential(inputs)


# class ConformerEncoder(nn.Module):
#     """
#     Conformer encoder first processes the input with a convolution subsampling layer and then
#     with a number of conformer blocks.

#     Args:
#         input_dim (int, optional): Dimension of input vector
#         encoder_dim (int, optional): Dimension of conformer encoder
#         num_layers (int, optional): Number of conformer blocks
#         num_attention_heads (int, optional): Number of attention heads
#         feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
#         conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
#         feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
#         attention_dropout_p (float, optional): Probability of attention module dropout
#         conv_dropout_p (float, optional): Probability of conformer convolution module dropout
#         conv_kernel_size (int or tuple, optional): Size of the convolving kernel
#         half_step_residual (bool): Flag indication whether to use half step residual or not

#     Inputs: inputs, input_lengths
#         - **inputs** (batch, time, dim): Tensor containing input vector
#         - **input_lengths** (batch): list of sequence input lengths

#     Returns: outputs, output_lengths
#         - **outputs** (batch, out_channels, time): Tensor produces by conformer encoder.
#         - **output_lengths** (batch): list of sequence output lengths
#     """
#     def __init__(
#             self,
#             input_dim: int = 80,
#             encoder_dim: int = 512,
#             num_layers: int = 17,
#             num_attention_heads: int = 8,
#             feed_forward_expansion_factor: int = 4,
#             conv_expansion_factor: int = 2,
#             input_dropout_p: float = 0.1,
#             feed_forward_dropout_p: float = 0.1,
#             attention_dropout_p: float = 0.1,
#             conv_dropout_p: float = 0.1,
#             conv_kernel_size: int = 31,
#             half_step_residual: bool = True,
#     ):
#         super(ConformerEncoder, self).__init__()
#         self.conv_subsample = Conv2dSubampling(in_channels=1, out_channels=encoder_dim)
#         self.input_projection = nn.Sequential(
#             Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim),
#             nn.Dropout(p=input_dropout_p),
#         )
#         self.layers = nn.ModuleList([ConformerBlock(
#             encoder_dim=encoder_dim,
#             num_attention_heads=num_attention_heads,
#             feed_forward_expansion_factor=feed_forward_expansion_factor,
#             conv_expansion_factor=conv_expansion_factor,
#             feed_forward_dropout_p=feed_forward_dropout_p,
#             attention_dropout_p=attention_dropout_p,
#             conv_dropout_p=conv_dropout_p,
#             conv_kernel_size=conv_kernel_size,
#             half_step_residual=half_step_residual,
#         ) for _ in range(num_layers)])

#     def count_parameters(self) -> int:
#         """ Count parameters of encoder """
#         return sum([p.numel() for p in self.parameters()])

#     def update_dropout(self, dropout_p: float) -> None:
#         """ Update dropout probability of encoder """
#         for name, child in self.named_children():
#             if isinstance(child, nn.Dropout):
#                 child.p = dropout_p

#     def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
#         """
#         Forward propagate a `inputs` for  encoder training.

#         Args:
#             inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
#                 `FloatTensor` of size ``(batch, seq_length, dimension)``.
#             input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

#         Returns:
#             (Tensor, Tensor)

#             * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
#                 ``(batch, seq_length, dimension)``
#             * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
#         """
#         outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
#         outputs = self.input_projection(outputs)

#         for layer in self.layers:
#             outputs = layer(outputs)

#         return outputs, output_lengths
