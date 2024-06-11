import torch
from torch import nn
import torch.nn.functional as F

from typing import List, Tuple, Dict, Union


def get_act(type):
    if type == 'relu':
        return nn.ReLU()
    elif type == 'gelu':
        return nn.GELU()
    else:
        raise NotImplementedError
    

def get_norm(type, **kwargs):
    if type == 'bn':
        return nn.BatchNorm2d(**kwargs)
    elif type == 'gn':
        return nn.GroupNorm(**kwargs)
    elif type == 'ln':
        return nn.GroupNorm(**kwargs)
    elif type == 'in':
        return nn.InstanceNorm2d(**kwargs)
    elif type == 'none':
        return nn.Identity()
    else:
        raise NotImplementedError
    
    
class ConvBlock(nn.Module):
    """conv-norm-relu"""
    def __init__(self, in_channels, out_channels, act, norm_config):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_config['type'], **norm_config['config']),
            get_act(act)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class TwoConvBlock(nn.Module):
    """TWO conv-norm-relu,
    conv1: in_channel -> out_channel
    conv2: out_channel -> out_channel
    """
    def __init__(self, in_channels, out_channels, act, norm_config):
        super(TwoConvBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, act, norm_config)
        self.conv2 = ConvBlock(out_channels, out_channels, act, norm_config)
    
    def forward(self, x):
        x = self.conv2(self.conv1(x))
        return x 
    
class SEBlock(nn.Module):
    """se block with parm: reduction"""
    def __init__(self, in_channels, reduction):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, 
                             in_channels // reduction, 
                             bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, 
                             in_channels, 
                             bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.global_avg_pool(x).view(b, c)
        # Excitation
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class TwoConvSEBlock(nn.Module):
    """TWO conv-norm-relu, the second conv block has a SE module
    conv1: in_channel -> out_channel
    conv2: out_channel -> out_channel"""
    def __init__(self, in_channels, out_channels, reduction, act, norm_config):
        super(TwoConvSEBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, act, norm_config)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm(norm_config['type'], **norm_config['config']),
            SEBlock(out_channels, reduction),
            get_act(act)
        )
    
    def forward(self, x):
        x = self.conv2(self.conv1(x))
        return x 
    
class TwoConvResidualBlock(nn.Module):
    """Two conv block with skip connection
    conv1: in_channel -> out_channel
    conv2: out_channel -> out_channel"""
    def __init__(self, in_channels, out_channels, act, norm_config):
        super(TwoConvResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, act, norm_config)
        self.conv2 = ConvBlock(out_channels, out_channels, act, norm_config)
    
        
        if in_channels!= out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 3, 1, 1) 
        else:
            self.skip_connection = nn.Identity()
        
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        # TODO: see NOTE
        # NOTE: some implementation split the conv block's relu to output
        return x + self.skip_connection(residual)
    

