import torch
from torch import nn
from einops import rearrange

import math

    
class VitStem(nn.Module):
    def __init__(self, 
                 seq_len: int,
                 embed_dim: int,
                 patch_size: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.dim = embed_dim
        
        self.positional_encoding = nn.Embedding(num_embeddings=seq_len, 
                                                embedding_dim=self.dim)
        self.patch_embeding = nn.Conv2d(3, 
                                        self.dim,
                                        kernel_size=patch_size, 
                                        stride=patch_size)

    
    def forward(self, x):
        assert len(x.shape) == 4
        # expect x of shape (b, 3, h, w)
        x = self.patch_embeding(x) # (b, dim, patch_h, patch_w)
        x = rearrange(x, 
                    'b d p_h p_w -> b (p_h p_w) d') # (b, num_patches, patch_dim)
        # add positional encoding
        patches = x + self.positional_encoding(torch.arange(self.seq_len).to(x.device))
        return patches
    
class TorchVit(nn.Module):
    def __init__(self, 
                 torch_transformer_encoder_config, 
                 img_size: int,
                 patch_size: int,
                 num_classes: int = 10) -> None:
        super().__init__()
        seq_len = (img_size // patch_size) ** 2
        dim = torch_transformer_encoder_config['layer_config']['d_model']
        
        self.patch_size = patch_size
        self.seq_len = seq_len
        
        self.cls_token = nn.Parameter(torch.randn(1, dim) * 0.02)
        
        self.vit_stem = VitStem(seq_len=seq_len, 
                                embed_dim=dim,
                                patch_size=patch_size, 
                                )
        
        encoder_layer = nn.TransformerEncoderLayer(**torch_transformer_encoder_config['layer_config'])
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         torch_transformer_encoder_config['num_layers'])
        
        self.cls_head = nn.Linear(dim, num_classes)
        
    
    def forward(self, x):
        # expect x in shape (b, c, h, w)
        assert len(x.shape) == 4
        b_s = x.shape[0]
        x = self.vit_stem(x) # (b, seq_len, dim)
        # cat cls token
        x = torch.cat([self.cls_token.unsqueeze(0).repeat(b_s, 1, 1), x], dim=1) # (b, seq_len+1, dim)
        x = self.transformer_encoder(x)
        img_embed = x[:, 0, :] # (b, dim)
        return self.cls_head(img_embed) # (b, num_classes)
    

class VitConvStem(nn.Module):
    def __init__(self, 
                 conv_channels: int,
                 embed_dim: int,
                 patch_size: int) -> None:
        super().__init__()
        self.dim = embed_dim
        
        stem_layers = []
        num_down_sample = int(math.log2(patch_size))
        in_channel = 3
        for _ in range(num_down_sample):
            stem_layers += [
                nn.Conv2d(in_channel, conv_channels, 3, 2, 1),
                nn.BatchNorm2d(conv_channels),
                nn.ReLU() 
            ]
            in_channel = conv_channels
        stem_layers.append(nn.Conv2d(conv_channels, embed_dim, 3, 1, 1))
        self.layers = nn.Sequential(*stem_layers)
            
    def forward(self, x):
        # expect x in shape (b, c, h, w)
        assert len(x.shape) == 4
        x = self.layers(x) # (b, embed, h', w')
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x



class TorchConvVit(nn.Module):
    def __init__(self, 
                 torch_transformer_encoder_config, 
                 img_size: int,
                 patch_size: int,
                 conv_channels: int,
                 num_classes: int = 10) -> None:
        super().__init__()
        seq_len = (img_size // patch_size) ** 2
        dim = torch_transformer_encoder_config['layer_config']['d_model']
        
        self.patch_size = patch_size
        self.seq_len = seq_len
        
        self.cls_token = nn.Parameter(torch.randn(1, dim) * 0.02)
        
        self.vit_stem = VitConvStem(conv_channels=conv_channels, 
                                embed_dim=dim,
                                patch_size=patch_size, 
                                )
        
        encoder_layer = nn.TransformerEncoderLayer(**torch_transformer_encoder_config['layer_config'])
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         torch_transformer_encoder_config['num_layers'])
        
        self.cls_head = nn.Linear(dim, num_classes)
        
    
    def forward(self, x):
        # expect x in shape (b, c, h, w)
        assert len(x.shape) == 4
        b_s = x.shape[0]
        x = self.vit_stem(x) # (b, seq_len, dim)
        # cat cls token
        x = torch.cat([self.cls_token.unsqueeze(0).repeat(b_s, 1, 1), x], dim=1) # (b, seq_len+1, dim)
        x = self.transformer_encoder(x)
        img_embed = x[:, 0, :] # (b, dim)
        return self.cls_head(img_embed) # (b, num_classes)