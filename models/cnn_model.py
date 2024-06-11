from typing import Union, List, Dict
from torch import nn
from models.get_blocks import get_block


class CnnModel(nn.Module):
    def __init__(self, 
                 channels: List[int], 
                 num_block_per_stage: Union[List[int], int], 
                 block_type: str,
                 act_type: str,
                 norm_type: str,
                 base_block_config: Dict,
                 in_channel: int = 3,
                 **kwargs):
        super(CnnModel, self).__init__()
        num_stages = len(channels) - 1
        if isinstance(num_block_per_stage, int):
            num_block_per_stage = [num_block_per_stage for _ in range(num_stages)]
        assert len(channels) == len(num_block_per_stage) + 1
        
        self.init_conv = nn.Conv2d(in_channel, channels[0], 3, 1, 1)
        
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels[-1], 10, 1, 1, 0),
        ) 
            
        
        self.encode_layers = nn.ModuleList()
        for i in range(num_stages):
            layer = nn.ModuleDict()
            res_layer = nn.Sequential()
            in_channels, out_channels = channels[i], channels[i + 1]
            
            base_block_config['in_channels'] = in_channels
            base_block_config['out_channels'] = out_channels
            base_block_config['act'] = act_type
            base_block_config['norm_config']['type'] = norm_type
            if norm_type == 'bn':
                base_block_config['norm_config']['config']['num_features'] = out_channels
            elif norm_type == 'gn':
                base_block_config['norm_config']['config']['num_channels'] = out_channels
                base_block_config['norm_config']['config']['num_groups'] = out_channels // kwargs['num_channels_per_gn_group']
            elif norm_type == 'ln':
                base_block_config['norm_config']['config']['num_channels'] = out_channels
                base_block_config['norm_config']['config']['num_groups'] = 1
            elif norm_type == 'in':
                base_block_config['norm_config']['config']['num_features'] = out_channels
            elif norm_type == 'none':
                pass
            else:
                raise NotImplementedError

            # first block
            res_layer.append(get_block(block_type, base_block_config))
            # next block
            base_block_config['in_channels'] = out_channels
            for _ in range(num_block_per_stage[i] - 1):
                res_layer.append(get_block(block_type, base_block_config))
            layer['layers'] = res_layer
            
            if i != num_stages - 1:
                layer['downsample'] = nn.MaxPool2d(2, 2)
            else:
                layer['downsample'] = nn.Identity()
            
            self.encode_layers.append(layer)
    
    def forward(self, x):
        x = self.init_conv(x)
        for l in self.encode_layers:
            x = l['layers'](x)
            x = l['downsample'](x)
        return self.cls_head(x).squeeze()