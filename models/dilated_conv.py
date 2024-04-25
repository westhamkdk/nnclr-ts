import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, train_data_L, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.final = final
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        
        # if final:
        #     self.projector = nn.Conv1d(in_channels, out_channels, train_data_L, padding=0)
        #     self.conv2 = nn.Conv1d(out_channels, out_channels, train_data_L, padding=0)

        if in_channels != out_channels or final:
            self.projector = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.projector = None
        
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        
        res = x+residual
        # if self.final:
        #     res = self.pool(x+residual)
            # res = self.linear(x+residual)

        return res

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size, train_data_L):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                train_data_L = train_data_L,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)
