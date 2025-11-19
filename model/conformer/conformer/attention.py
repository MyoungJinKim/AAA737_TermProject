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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .embedding import RelPositionalEncoding
from .modules import Linear

class RelativeMultiHeadAttention(nn.Module):
    """
    상대적 위치 인코딩(Relative Positional Encoding)을 적용한 Multi-head Attention.
    Transformer-XL에서 제안된 방식으로, 절대 위치가 아니라 "상대적 거리"에 기반해
    어텐션 점수를 계산함으로써 더 긴 문맥 일반화 능력을 얻는 기법.

    query, key, value 외에 pos_embedding(상대적 위치 임베딩)을 추가로 사용하며
    content-based attention + position-based attention 두 가지 점수를 결합한다.
    """
    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super(RelativeMultiHeadAttention, self).__init__()

        # head 크기 = d_model / num_heads
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)  # 스케일링 값

        # Q, K, V, Positional Embedding 선형 변환
        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)

        # Transformer-XL에서 사용하는 learnable bias(u, v)
        # u_bias: content-based bias
        # v_bias: position-based bias
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(d_model, d_model)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            pos_embedding: Tensor,
            mask: Optional[Tensor] = None,
    ) -> Tensor:

        batch_size = value.size(0)

        # Q: (B, T, H, D_head)
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)

        # K, V: (B, H, T, D_head)로 reshape 후 head 차원 앞으로 이동
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        # pos embedding: (B, T, H, D_head)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        # ─────────────────────────────────────────────
        # 1) Content-based attention score
        #    (Q + u) · K^T
        # ─────────────────────────────────────────────
        content_score = torch.matmul(
            (query + self.u_bias).transpose(1, 2),   # (B, H, T1, D)
            key.transpose(2, 3)                      # (B, H, D, T2)
        )  # → (B, H, T1, T2)

        # ─────────────────────────────────────────────
        # 2) Position-based attention score
        #    (Q + v) · R^T 에 해당하는 항
        # ─────────────────────────────────────────────
        pos_score = torch.matmul(
            (query + self.v_bias).transpose(1, 2),   # (B, H, T1, D)
            pos_embedding.permute(0, 2, 3, 1)        # (B, H, D, T2)
        )
        # Transformer-XL에서 제안한 relative shift trick 적용
        pos_score = self._relative_shift(pos_score)

        # 전체 score 결합 + scaling
        score = (content_score + pos_score) / self.sqrt_dim

        # ─────────────────────────────────────────────
        # Mask 적용 (padding mask 또는 causal mask)
        # ─────────────────────────────────────────────
        if mask is not None:
            mask = mask.unsqueeze(1)  # head 차원에 broadcast
            score.masked_fill_(mask, -1e9)

        # softmax로 attention weight 계산
        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        # context 계산: attention * value
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        """
        Transformer-XL의 핵심 기법인 relative shift 구현.
        위치 인코딩 행렬을 한 칸 shift하여 (i, j)에서의 상대적 거리(i-j)에 맞는
        상대적 위치 점수만 남기도록 재배치하는 과정.
        """
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()

        # 맨 앞에 0을 pad하여 shift 효과 준비
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        # reshape을 통해 실제 shift 실행
        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)

        # 필요한 부분만 잘라냄
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score


class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Conformer에서 사용하는 Relative Multi-head Self Attention(MHSA) 구조.
    Transformer-XL의 상대적 위치 인코딩을 통합하여 입력 길이가 달라져도
    모델이 더 robust 하도록 설계됨.
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()

        # 상대적 위치 인코딩 생성 모듈
        self.positional_encoding = RelPositionalEncoding(d_model)

        # Pre-Norm 구조 적용
        self.layer_norm = nn.LayerNorm(d_model)

        # 상대적 위치 인코딩이 적용된 Multi-head Attention
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
        batch_size = inputs.size(0)

        # 입력과 길이에 맞는 상대적 위치 인코딩 생성
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)  # 배치 크기만큼 반복

        # LayerNorm + MHSA
        inputs = self.layer_norm(inputs)
        outputs = self.attention(
            inputs, inputs, inputs,
            pos_embedding=pos_embedding,
            mask=mask,
        )

        return self.dropout(outputs)


# class RelativeMultiHeadAttention(nn.Module):
#     """
#     Multi-head attention with relative positional encoding.
#     This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

#     Args:
#         d_model (int): The dimension of model
#         num_heads (int): The number of attention heads.
#         dropout_p (float): probability of dropout

#     Inputs: query, key, value, pos_embedding, mask
#         - **query** (batch, time, dim): Tensor containing query vector
#         - **key** (batch, time, dim): Tensor containing key vector
#         - **value** (batch, time, dim): Tensor containing value vector
#         - **pos_embedding** (batch, time, dim): Positional embedding tensor
#         - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

#     Returns:
#         - **outputs**: Tensor produces by relative multi head attention module.
#     """
#     def __init__(
#             self,
#             d_model: int = 512,
#             num_heads: int = 16,
#             dropout_p: float = 0.1,
#     ):
#         super(RelativeMultiHeadAttention, self).__init__()
#         assert d_model % num_heads == 0, "d_model % num_heads should be zero."
#         self.d_model = d_model
#         self.d_head = int(d_model / num_heads)
#         self.num_heads = num_heads
#         self.sqrt_dim = math.sqrt(self.d_head)

#         self.query_proj = Linear(d_model, d_model)
#         self.key_proj = Linear(d_model, d_model)
#         self.value_proj = Linear(d_model, d_model)
#         self.pos_proj = Linear(d_model, d_model, bias=False)

#         self.dropout = nn.Dropout(p=dropout_p)
#         self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
#         self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
#         torch.nn.init.xavier_uniform_(self.u_bias)
#         torch.nn.init.xavier_uniform_(self.v_bias)

#         self.out_proj = Linear(d_model, d_model)

#     def forward(
#             self,
#             query: Tensor,
#             key: Tensor,
#             value: Tensor,
#             pos_embedding: Tensor,
#             mask: Optional[Tensor] = None,
#     ) -> Tensor:
#         batch_size = value.size(0)

#         query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
#         key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
#         value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
#         pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

#         content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
#         pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
#         pos_score = self._relative_shift(pos_score)

#         score = (content_score + pos_score) / self.sqrt_dim

#         if mask is not None:
#             mask = mask.unsqueeze(1)
#             score.masked_fill_(mask, -1e9)

#         attn = F.softmax(score, -1)
#         attn = self.dropout(attn)

#         context = torch.matmul(attn, value).transpose(1, 2)
#         context = context.contiguous().view(batch_size, -1, self.d_model)

#         return self.out_proj(context)

#     def _relative_shift(self, pos_score: Tensor) -> Tensor:
#         batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
#         zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
#         padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

#         padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
#         pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

#         return pos_score


# class MultiHeadedSelfAttentionModule(nn.Module):
#     """
#     Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
#     the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
#     module to generalize better on different input length and the resulting encoder is more robust to the variance of
#     the utterance length. Conformer use prenorm residual units with dropout which helps training
#     and regularizing deeper models.

#     Args:
#         d_model (int): The dimension of model
#         num_heads (int): The number of attention heads.
#         dropout_p (float): probability of dropout

#     Inputs: inputs, mask
#         - **inputs** (batch, time, dim): Tensor containing input vector
#         - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

#     Returns:
#         - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
#     """
#     def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
#         super(MultiHeadedSelfAttentionModule, self).__init__()
#         self.positional_encoding = RelPositionalEncoding(d_model)
#         self.layer_norm = nn.LayerNorm(d_model)
#         self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
#         self.dropout = nn.Dropout(p=dropout_p)

#     def forward(self, inputs: Tensor, mask: Optional[Tensor] = None):
#         batch_size = inputs.size(0)
#         pos_embedding = self.positional_encoding(inputs)
#         pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

#         inputs = self.layer_norm(inputs)
#         outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

#         return self.dropout(outputs)
