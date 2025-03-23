#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2023/6/12 5:53
@File:          channel_attention.py
'''

import math
from typing import Optional, Callable, Union, Tuple, List
from functools import partial
from torch import Tensor
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def unfold2d(
        inputs: Tensor,
        kernel_size: int | Tuple[int, int],
        dilation: int | Tuple[int, int] = 1,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] | Tuple[int, int, int, int] | None = None,
        padding_value: float = 0.) -> Tensor:
    if padding is None:
        if isinstance(kernel_size, int):
            padding_ = (kernel_size // 2,) * 4
        else:
            padding_ = (kernel_size[0] // 2, kernel_size[0] // 2,
                        kernel_size[1] // 2, kernel_size[1] // 2)
    elif isinstance(padding, int):
        padding_ = (padding,) * 4
    else:
        if len(padding) == 2:
            padding_ = (padding[0], padding[0], padding[1], padding[1])
        else:
            padding_ = padding

    x = F.pad(inputs, padding_, value=padding_value)
    x = F.unfold(x, kernel_size, dilation=dilation, stride=stride)
    return x

class LogsumexpPool2d(nn.Module):
    _N = 2

    # a custom module that implements logsumexp pooling over a 2D input
    def __init__(self, kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 temperature: float = 1e-12) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * LogsumexpPool2d._N

        if stride is None:
            stride = kernel_size
        else:
            if isinstance(stride, int):
                stride = (stride,) * LogsumexpPool2d._N

        if isinstance(padding, int):
            padding = (padding,) * LogsumexpPool2d._N

        if isinstance(dilation, int):
            dilation = (dilation,) * LogsumexpPool2d._N

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.temperature = temperature

    def forward(self, inputs: Tensor) -> Tensor:
        b, c, h, w = inputs.size()
        i = inputs.view(b * c, 1, h, w)
        x = unfold2d(i, self.kernel_size, dilation=self.dilation, padding=self.padding,
                     stride=self.stride, padding_value=-math.inf)  # shape: (N * C, 1 * kernel_size * kernel_size, L)
        x = torch.logsumexp(x / self.temperature, dim=1) * self.temperature  # shape: (N * C, L)
        out = x.view(b, c, *self.output_size((h, w)))  # shape: (N, C, H_out, W_out)
        return out

    def output_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        H = int((size[0] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        W = int((size[1] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        return H, W

    def extra_repr(self) -> str:
        return ('{kernel_size}, '
                'stride = {stride}, '
                'padding = {padding}, '
                'dilation = {dilation}, '
                'temperature = {temperature}'.format(**self.__dict__))

class GSACA(nn.Module):
    '''引自论文Improved channel attention methods via hierarchical pooling and reducing information loss
    '''
    def __init__(self,
                 inplanes: int,
                 grouped_width: Optional[int] = 32,
                 groups: Optional[int] = None,
                 default_groups: int = 4,
                 dropout_rate: float = 0.2,
                 conv_layer: Optional[Callable[..., nn.Module]] = partial(nn.Conv2d, bias=False),
                 scale_layer: Optional[Callable[..., nn.Module]] = nn.Sigmoid) -> None:
        if conv_layer is None:
            conv_layer = partial(nn.Conv2d, bias=False)
        if scale_layer is None:
            scale_layer = nn.Sigmoid

        if grouped_width is not None:
            g = inplanes // grouped_width or default_groups
        else:
            g = groups or default_groups

        super().__init__()

        self.hierarchical_pooling = nn.Sequential(
            LogsumexpPool2d((3, 3), padding=(1, 1)),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.conv1 = conv_layer(inplanes, inplanes * 2, 1, groups=g)
        self.conv2 = conv_layer(inplanes, inplanes, 1, groups=g)
        self.scale = scale_layer()
        self.dropout = nn.Dropout(p=dropout_rate, inplace=True)
        self.groups = g
        self.inplanes = inplanes

    def forward(self, inputs: Tensor) -> Tensor:
        # Squeeze: 采用层次池化
        z = self.hierarchical_pooling(inputs)

        # Excitation
        '''
        通过分组-组交互-聚合建立通道关系，带来两个好处：
        1）信息减损（不降维）。
        2）所有通道分量都参与了每个通道关系分量的计算。
        '''
        x = self.conv1(z)
        x, g = x[:, :self.inplanes, :, :], torch.special.expit(self.dropout(x[:, self.inplanes:, :, :]))
        x = z * (1 - g) + x * g
        x = self.channel_shuffle(x)
        x = self.conv2(x)
        x = self.scale(x)

        # Scale
        y = inputs * x.expand_as(inputs)
        return y

    def channel_shuffle(self, inputs: Tensor) -> Tensor:
        b, c = inputs.size()[:2]
        x = inputs.view(b, self.groups, c // self.groups, 1, 1)
        x = x.transpose(1, 2).contiguous().view(b, c, 1, 1)
        return x

class MECA(nn.Module):
    '''引自论文Improved channel attention methods via hierarchical pooling and reducing information loss
    '''
    def __init__(self,
                 inplanes: int,
                 reduction: float = 0.0625,
                 rplanes: Optional[int] = None,
                 grouped_width: Optional[int] = 32,
                 groups: Optional[int] = None,
                 default_groups: int = 4,
                 p: float = 0.5,
                 conv_layer: Optional[Callable[..., nn.Module]] = partial(nn.Conv2d, bias=False),
                 scale_layer: Optional[Callable[..., nn.Module]] = nn.Sigmoid) -> None:
        if conv_layer is None:
            conv_layer = partial(nn.Conv2d, bias=False)
        if scale_layer is None:
            scale_layer = nn.Sigmoid

        rplanes = rplanes or max(int(inplanes * reduction), 1)

        if grouped_width is not None:
            g = inplanes // grouped_width or default_groups
        else:
            g = groups or default_groups

        super().__init__()

        self.hierarchical_pooling = nn.Sequential(
            LogsumexpPool2d((3, 3), padding=(1, 1)),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.conv1 = conv_layer(inplanes, rplanes * 2, 1)
        self.conv2 = conv_layer(rplanes, inplanes, 1)
        self.gconv = conv_layer(inplanes, inplanes, 1, groups=g)
        self.scale = scale_layer()
        self.rplanes = rplanes
        self.p = p

    def forward(self, inputs: Tensor) -> Tensor:
        # Squeeze: 采用层次池化
        x = self.hierarchical_pooling(inputs)

        # MLP excitation
        z = self.conv1(x)
        z1, g = z[:, :self.rplanes, :, :], torch.special.expit(z[:, self.rplanes:, :, :])
        x1 = z1 * g
        x1 = self.scale(self.conv2(x1))

        # gconv excitation
        x2 = self.scale(self.gconv(x))

        # Random average fusion
        if self.training:
            lamb = np.random.choice((0, 1), size=1, p=(self.p, 1 - self.p))[0]
            x = x1 * (1 - lamb) + x2 * lamb
        else:
            x = (x1 + x2) / 2

        # Scale
        y = inputs * x.expand_as(inputs)
        return y