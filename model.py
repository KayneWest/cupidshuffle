
"""
code credit:
https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/shufflenetv2.py
https://github.com/mulinmeng/Shuffle-Transformer

"""



import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import torch
import torch.nn as nn
import torch.nn.functional as F

def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)

def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels // groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x

class LazyPatchEmbedding(nn.Module):
    def __init__(self, in_channels, inter_channel=32, out_channels=48):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU6(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channel, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0., stride=False):
        super().__init__()
        self.stride = stride
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, window_size=1, shuffle=False, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., relative_pos_embedding=False):
        super().__init__()
        self.num_heads = num_heads
        self.relative_pos_embedding = relative_pos_embedding
        head_dim = dim // self.num_heads
        self.ws = window_size
        self.shuffle = shuffle

        self.scale = qk_scale or head_dim ** -0.5

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)
            print('The relative_pos_embedding is used')

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)

        if self.shuffle:
            q, k, v = rearrange(qkv, 'b (qkv h d) (ws1 hh) (ws2 ww) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads, qkv=3, ws1=self.ws, ws2=self.ws)
        else:
            q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        out = attn @ v

        if self.shuffle:
            out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (ws1 hh) (ws2 ww)', h=self.num_heads, b=b, hh=h//self.ws, ws1=self.ws, ws2=self.ws)
        else:
            out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b, hh=h//self.ws, ws1=self.ws, ws2=self.ws)
 
        out = self.proj(out)
        out = self.proj_drop(out)

        return out

class TBlock(nn.Module):
    def __init__(self, dim, out_dim, num_heads, window_size=1, shuffle=False, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, stride=False, relative_pos_embedding=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, window_size=window_size, shuffle=shuffle, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, relative_pos_embedding=relative_pos_embedding)
        self.local = nn.Conv2d(dim, dim, window_size, 1, window_size//2, groups=dim, bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer, drop=drop, stride=stride)
        self.norm3 = norm_layer(dim)
        print("input dim={}, output dim={}, stride={}, expand={}, num_heads={}".format(dim, out_dim, stride, shuffle, num_heads))

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.local(self.norm2(x)) # local connection
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x

class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )
        else:
            self.shortcut = nn.Sequential()

            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 1 and self.out_channels == self.in_channels:
            shortcut, residual = channel_split(x, int(self.in_channels / 2))
        else:
            shortcut = x
            residual = x

        shortcut = self.shortcut(shortcut)
        residual = self.residual(residual)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)
        return x

class CupidShuffle(nn.Module):

    def __init__(self, ratio=1, start_channels = 24, class_num=100, use_block=True, repeats=[3, 7, 3], token_dim=32, embed_dim=96, has_pos_embed=False):
        super().__init__()

        self.use_block = use_block
        
        if ratio == 0.5:
            out_channels = [48, 96, 192, 1024]
        elif ratio == 1:
            out_channels = [116, 232, 464, 1024]
        elif ratio == 1.5:
            out_channels = [176, 352, 704, 1024]
        elif ratio == 2:
            out_channels = [244, 488, 976, 2048]
        else:
            ValueError('unsupported ratio number')

        self.stage2 = self._make_stage(start_channels, out_channels[0], repeats[0])
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], repeats[1])
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], repeats[2])
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(out_channels[3], class_num)
        
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        num_patches = (start_channels * start_channels) // 16

        self.to_token = LazyPatchEmbedding(in_channels=3, inter_channel=token_dim, out_channels=start_channels)
        
        self.has_pos_embed = has_pos_embed

        if self.has_pos_embed:
            self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=embed_dim), requires_grad=False)
            self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_block:
            self.block = TBlock(start_channels, start_channels, 1, shuffle=False, relative_pos_embedding=True)


    def forward(self, x):
        x = self.to_token(x)
        b, c, h, w = x.shape

        if self.has_pos_embed:
            x = x + self.pos_embed.view(1, h, w, c).permute(0, 3, 1, 2)
            x = self.pos_drop(x)
        if self.use_block:
            x = self.block(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = []
  
        layers.append(ShuffleUnit(in_channels, out_channels, 2))

        while repeat:
            layers.append(ShuffleUnit(out_channels, out_channels, 1))
            repeat -= 1

        return nn.Sequential(*layers)
