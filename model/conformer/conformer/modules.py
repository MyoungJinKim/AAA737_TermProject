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
import torch.nn.init as init
from torch import Tensor


class ResidualConnectionModule(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)


class Linear(nn.Module):
    """
    torch.nn.Linear를 감싸는(wrapper) 클래스.
    주요 차이점:
        - Xavier 초기화를 기본 적용 (weight)
        - bias는 0으로 초기화

    역할:
        입력 x의 마지막 차원(in_features)을 out_features로 선형 매핑하는 모듈.
        즉,
            (B, T, in_features) → Linear → (B, T, out_features)

    선형 변환 식:
        y = x * W^T + b
        - x: (..., in_features)
        - W: (out_features, in_features)
        - b: (out_features)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()

        # nn.Linear는 입력 차원의 마지막 축(in_features)을 out_features로 매핑함
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Xavier 초기화: 입력·출력의 분산을 균형있게 조정 (딥러닝에서 매우 일반적)
        init.xavier_uniform_(self.linear.weight)

        # bias는 0으로 초기화
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward 과정:

        입력 x 차원 예시:
            - (B, T, D_in)  →   NLP/ASR에서 일반 형태
            - (B, D_in)
            - (..., D_in)

        Linear는 마지막 차원(D_in)만 변환하므로:
            출력 차원: (..., out_features)

        예:
            x: (B, T, in_features)
            y = Linear(x)
               → (B, T, out_features)
        """
        return self.linear(x)


class View(nn.Module):
    """ Wrapper class of torch.view() for Sequential module. """
    def __init__(self, shape: tuple, contiguous: bool = False):
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, x: Tensor) -> Tensor:
        if self.contiguous:
            x = x.contiguous()

        return x.view(*self.shape)


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)
