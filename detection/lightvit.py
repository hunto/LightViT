import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from timm.models.layers import DropPath, trunc_normal_, drop_path

from mmdet.models.builder import BACKBONES
from mmcv.runner import (auto_fp16, force_fp32,)
from mmcv.runner import BaseModule


class ConvStem(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_eval=False):
        super().__init__()

        self.patch_size = patch_size
        self.norm_eval = norm_eval

        stem_dim = embed_dim // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_dim, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3,
                      groups=stem_dim, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3,
                      groups=stem_dim, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            nn.Conv2d(stem_dim, stem_dim, kernel_size=3,
                      groups=stem_dim, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(stem_dim, embed_dim,
                              kernel_size=3,
                              stride=2, padding=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        stem = self.stem(x)
        x = self.proj(stem)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H, W)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.25, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean(1, keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]
        s_attn = self.spatial_select(torch.cat([x_local, x_global.expand(-1, x.shape[1], -1)], dim=-1))
        s_attn = self.gate_fn(s_attn)  # [B, N, 1]

        attn = c_attn * s_attn  # [B, N, C]
        return ori_x * attn


class BiAttnMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.attn = BiAttn(out_features)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.attn(x)
        x = self.drop(x)
        return x


def window_reverse(
        windows: torch.Tensor,
        original_size,
        window_size=(7, 7)
) -> torch.Tensor:
    """ Reverses the window partition.
    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0] * window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)
    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, original_size[0] * original_size[1], C].
    """
    # Get height and width
    H, W = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 1, 3, 2, 4, 5).reshape(B, H * W, -1)
    return output


def get_relative_position_index(
        win_h: int,
        win_w: int
) -> torch.Tensor:
    """ Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.
    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.
    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


class Attention(nn.Module):
    def __init__(self, dim, num_tokens=1, num_heads=8, window_size=7, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.num_tokens = num_tokens
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        # positional embedding
        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward_global_aggregation(self, q, k, v):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward_local(self, q, k, v, H, W):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        B, num_heads, N, C = q.shape
        ws = self.window_size
        h_group, w_group = H // ws, W // ws

        # partition to windows
        q = q.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        q = q.view(-1, num_heads, ws*ws, C)
        k = k.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        k = k.view(-1, num_heads, ws*ws, C)
        v = v.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        v = v.view(-1, num_heads, ws*ws, v.shape[-1])

        attn = (q @ k.transpose(-2, -1)) * self.scale
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws*ws, -1)

        # reverse
        x = window_reverse(x, (H, W), (ws, ws))
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward(self, x, H, W):
        B, N, C = x.shape
        NC = self.num_tokens
        # pad
        x_img, x_global = x[:, NC:], x[:, :NC]
        x_img = x_img.view(B, H, W, C)
        pad_l = pad_t = 0
        ws = self.window_size
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        x_img = F.pad(x_img, (0, 0, pad_l, pad_r, pad_t, pad_b))
        Hp, Wp = x_img.shape[1], x_img.shape[2]
        x_img = x_img.view(B, -1, C)
        x = torch.cat([x_global, x_img], dim=1)

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NC:], k[:, :, NC:], v[:, :, NC:]
        q_cls, _, _ = q[:, :, :NC], k[:, :, :NC], v[:, :, :NC]

        # local window attention
        x_img = self.forward_local(q_img, k_img, v_img, Hp, Wp)
        # restore to the original size
        x_img = x_img.view(B, Hp, Wp, -1)[:, :H, :W].reshape(B, H*W, -1)
        q_img = q_img.reshape(B, self.num_heads, Hp, Wp, -1)[:, :, :H, :W].reshape(B, self.num_heads, H*W, -1)
        k_img = k_img.reshape(B, self.num_heads, Hp, Wp, -1)[:, :, :H, :W].reshape(B, self.num_heads, H*W, -1)
        v_img = v_img.reshape(B, self.num_heads, Hp, Wp, -1)[:, :, :H, :W].reshape(B, self.num_heads, H*W, -1)

        # global aggregation
        x_cls = self.forward_global_aggregation(q_cls, k_img, v_img)
        k_cls, v_cls = self.kv_global(x_cls).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)

        # gloal broadcast
        x_img = x_img + self.forward_global_broadcast(q_img, k_cls, v_cls)

        x = torch.cat([x_cls, x_img], dim=1)
        x = self.proj(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, num_tokens=1, window_size=7, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention=Attention, last_block=False):
        super().__init__()
        self.last_block = last_block
        self.norm1 = norm_layer(dim)
        self.attn = attention(dim, num_heads=num_heads, num_tokens=num_tokens, window_size=window_size, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = BiAttnMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        if self.last_block:
            # ignore unused global tokens in downstream tasks
            x = x[:, -H*W:]
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ResidualMergePatch(nn.Module):
    def __init__(self, dim, out_dim, num_tokens=1):
        super().__init__()
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim, bias=False)
        # use MaxPool3d to avoid permutations
        self.maxp = nn.MaxPool3d((2, 2, 1), (2, 2, 1))
        self.res_proj = nn.Linear(dim, out_dim, bias=False)

    def forward(self, x, H, W):
        global_token, x = x[:, :self.num_tokens].contiguous(), x[:, self.num_tokens:].contiguous()
        B, L, C = x.shape

        x = x.view(B, H, W, C)
        # pad
        pad_l = pad_t = 0
        pad_r = (2 - W % 2) % 2
        pad_b = (2 - H % 2) % 2
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        res = self.res_proj(self.maxp(x).view(B, -1, C))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        x = x + res
        global_token = self.proj(self.norm2(global_token))
        x = torch.cat([global_token, x], 1)
        return x, (math.ceil(H / 2), math.ceil(W / 2))


class LightViT(BaseModule):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[32, 64, 160, 256], num_layers=[2, 2, 2, 2],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], num_tokens=8, window_size=7, neck_dim=1280, qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=ConvStem, norm_layer=None,
                 act_layer=None, weight_init='', out_indices=(0, 1, 2, 3), stem_norm_eval=False, init_cfg=None):
        super().__init__(init_cfg)
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_tokens = num_tokens
        self.mlp_ratios = mlp_ratios
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.out_indices = out_indices
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0], norm_eval=stem_norm_eval)

        self.global_token = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dims[0]))

        stages = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]  # stochastic depth decay rule
        for stage, (embed_dim, num_layer, num_head, mlp_ratio) in enumerate(zip(embed_dims, num_layers, num_heads, mlp_ratios)):
           blocks = []
           if stage > 0:
               # downsample
               blocks.append(ResidualMergePatch(embed_dims[stage-1], embed_dim, num_tokens=num_tokens))
           blocks += [
               Block(
                   dim=embed_dim, num_heads=num_head, num_tokens=num_tokens, window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                   attn_drop=attn_drop_rate, drop_path=dpr[sum(num_layers[:stage]) + i], norm_layer=norm_layer, act_layer=act_layer, attention=Attention,
                   last_block=(stage==len(embed_dims)-1 and i == num_layer - 1))
               for i in range(num_layer)]
           blocks = nn.Sequential(*blocks)
           stages.append(blocks)
        self.stages = nn.Sequential(*stages)
        
        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(embed_dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward_features(self, x):
        outputs = []
        x, (H, W) = self.patch_embed(x)
        x_out = x.view(-1, H, W, x.shape[-1]).permute(0, 3, 1, 2).contiguous()
        global_token = self.global_token.expand(x.shape[0], -1, -1)
        x = torch.cat((global_token, x), dim=1)

        for i_stage, stage in enumerate(self.stages):
            for block in stage:
                if isinstance(block, ResidualMergePatch):
                    x, (H, W) = block(x, H, W)
                elif isinstance(block, Block):
                    x = block(x, H, W)
                else:
                    x = block(x)

            if i_stage in self.out_indices:
                norm_layer = getattr(self, f'norm{i_stage}')
                x_out = norm_layer(x[:, -H*W:])
                x_out = x_out.view(-1, H, W, self.embed_dims[i_stage]).permute(0, 3, 1, 2).contiguous()
                outputs.append(x_out)
        return tuple(outputs)

    @auto_fp16()
    def forward(self, x):
        x = self.forward_features(x)
        return x


@BACKBONES.register_module()
class lightvit_tiny(LightViT):

    def __init__(self, **kwargs):
        model_kwargs = dict(patch_size=8, embed_dims=[64, 128, 256], num_layers=[2, 6, 6],
                            num_heads=[2, 4, 8, ], mlp_ratios=[8, 4, 4], num_tokens=512, drop_path_rate=0.1)
        model_kwargs.update(kwargs)
        super().__init__(**model_kwargs)


@BACKBONES.register_module()
class lightvit_small(LightViT):

    def __init__(self, **kwargs):
        model_kwargs = dict(patch_size=8, embed_dims=[96, 192, 384], num_layers=[2, 6, 6],
                            num_heads=[3, 6, 12, ], mlp_ratios=[8, 4, 4], num_tokens=16, drop_path_rate=0.1)
        model_kwargs.update(kwargs)
        super().__init__(**model_kwargs)


@BACKBONES.register_module()
class lightvit_base(LightViT):

    def __init__(self, **kwargs):
        model_kwargs = dict(patch_size=8, embed_dims=[128, 256, 512], num_layers=[3, 8, 6],
                            num_heads=[4, 8, 16, ], mlp_ratios=[8, 4, 4], num_tokens=24, drop_path_rate=0.1)
        model_kwargs.update(kwargs)
        super().__init__(**model_kwargs)
