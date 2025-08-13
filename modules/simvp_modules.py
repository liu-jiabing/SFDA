import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
from timm.models.layers import DropPath, trunc_normal_


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class MDP(nn.Module):
    """保留完整空间注意力版：仅移除通道注意力"""
    def __init__(self, in_channels, max_levels=3):
        super().__init__()
        self.in_channels = in_channels
        self.max_levels = max_levels

        self.conv_diff = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=3, padding=1,
                                   groups=in_channels)

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

       
        group_num = min(32, in_channels)  
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * max_levels, in_channels,
                      kernel_size=1,
                      groups=group_num,  
                      bias=False),
            nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels),
            nn.GELU()
        )

        self.level_weights = nn.Parameter(torch.ones(1, max_levels, 1, 1))
        self.softmax = nn.Softmax(dim=1)

    def sqrt_abs_diff(self, x, y):
        return torch.sqrt(torch.abs(x - y) + 1e-6)

    def forward(self, *diff_inputs):
        diffs = []
        for i in range(len(diff_inputs)):
            for j in range(i + 1, len(diff_inputs)):
                diffs.append(self.sqrt_abs_diff(diff_inputs[i], diff_inputs[j]))

        active_levels = min(len(diffs), self.max_levels)
        if active_levels == 0:
            return torch.zeros_like(diff_inputs[0])

        pyramid = []
        for i in range(active_levels):
            if i == 0:
                resized_diff = diffs[i]
            else:
                resized_diff = F.avg_pool2d(diffs[i], kernel_size=2 ** i)
            processed = self.conv_diff(resized_diff)
            pyramid.append(processed)

        weighted_features = []
        norm_weights = self.softmax(self.level_weights)[:, :active_levels]

        base_size = pyramid[0].shape[2:]
        for i, (feat, weight) in enumerate(zip(pyramid, norm_weights.split(1, dim=1))):
            avg_pool = torch.mean(feat, dim=1, keepdim=True)
            max_pool, _ = torch.max(feat, dim=1, keepdim=True)
            spatial_attn = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))

            if feat.shape[2:] != base_size:
                spatial_attn = F.interpolate(spatial_attn, size=base_size, mode='bilinear')
                feat = F.interpolate(feat, size=base_size, mode='bilinear')
            weighted = spatial_attn * feat * weight
            weighted_features.append(weighted)

       
        if len(weighted_features) < self.max_levels:
            dummy = torch.zeros_like(weighted_features[0])
            weighted_features += [dummy] * (self.max_levels - len(weighted_features))

        fused = torch.cat(weighted_features, dim=1)
        return self.fusion_conv(fused)




class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


class ActivationLinear(nn.Module):
    def __init__(self, in_dim, out_dim, use_gelu=False, bias=True):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim, bias=bias)]
        if use_gelu:
            layers.append(nn.GELU())
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
            
        return self.proj(x)


class MP(nn.Module):
    """优化版多头KAN：增加参数化Krylov子空间和残差连接"""
    def __init__(self, dim, krylov_dim, heads=4, order=3, use_bias=True):
        super().__init__()
        self.heads = heads
        self.head_dim = krylov_dim // heads
        self.scale = self.head_dim ** -0.5
        self.order = order
        self.q_projs = nn.ModuleList([
            ActivationLinear(dim, krylov_dim, use_gelu=(i > 0), bias=use_bias)
            for i in range(order)
        ])
        self.k_projs = nn.ModuleList([
            ActivationLinear(dim, krylov_dim, use_gelu=(i > 0), bias=use_bias)
            for i in range(order)
        ])
        self.v_projs = nn.ModuleList([
            ActivationLinear(dim, krylov_dim, use_gelu=(i > 0), bias=use_bias)
            for i in range(order)
        ])

        self.to_out = nn.Sequential(
            nn.Linear(krylov_dim * order, dim, bias=use_bias),
            nn.LayerNorm(dim) if use_bias else nn.Identity()
        )

        self.order_weights = nn.Parameter(torch.ones(order, 1, 1, 1))  # 每阶一个权重参数

    def forward(self, x):
        B, N, C = x.shape
        outputs = []

        for i in range(self.order):
            q = self.q_projs[i](x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
            k = self.k_projs[i](x).view(B, N, self.heads, self.head_dim).transpose(1, 2)
            v = self.v_projs[i](x).view(B, N, self.heads, self.head_dim).transpose(1, 2)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, -1)

            outputs.append(out * self.order_weights[i])

        out = torch.cat(outputs, dim=-1)
        return self.to_out(out)


class GroupedMLP(nn.Module):
    def __init__(self, dim, mlp_hidden_dim, groups=4, drop=0.):
        super().__init__()
        self.groups = groups
        self.group_dim = dim // groups
        self.group_hidden_dim = mlp_hidden_dim // groups

        self.linear1 = nn.Linear(self.group_dim, self.group_hidden_dim, bias=True)
        self.linear2 = nn.Linear(self.group_hidden_dim, self.group_dim, bias=True)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, N, self.groups, self.group_dim)
        x = x.permute(0, 2, 1, 3).contiguous().view(B * self.groups, N, self.group_dim)

        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        x = x.view(B, self.groups, N, self.group_dim).permute(0, 2, 1, 3).contiguous()
        return x.view(B, N, C)


class MPA(nn.Module):
    """增强版MultiKAN层：添加门控机制和FFN"""
    def __init__(self, dim, krylov_dim, order=3, rank=8, heads=4,
                 mlp_ratio=2.0, drop=0., drop_path=0.1, use_gate=True):
        super().__init__()

        self.multi_head_kan = MP(
            dim=dim,
            krylov_dim=krylov_dim,
            heads=heads,
            order=order
        )

        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(dim, 1),
                nn.Sigmoid()
            )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GroupedMLP(dim, mlp_hidden_dim, groups=4, drop=drop)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
   
        residual = x
        x_kan = self.norm1(x)
        x_kan = self.multi_head_kan(x_kan)

        if self.use_gate:
            gate_value = self.gate(x_kan)
            x_kan = x_kan * gate_value

        x = residual + self.drop_path(x_kan)

    
        residual = x
      
        x_mlp = self.norm2(x)
        x_mlp = self.mlp(x_mlp)
        x = residual + self.drop_path(x_mlp)
    
        return x



class ViTSubBlock(nn.Module):
    """完整ViT子模块（改进4：共享卷积权重 + 高效MLP）"""
    def __init__(self, dim, mlp_ratio=2., drop=0., drop_path=0.1, max_diff_levels=3, groups=2, krylov_dim=32, order=3, rank=8, heads=4):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)

        # 使用升级版的KAN
        self.mlp = MPA(
            dim=dim,
            krylov_dim=krylov_dim,
            order=order,
            rank=rank,
            heads=heads
        )

        # 改进4：共享基础卷积
        self.base_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.dilations = [1, 2, 3]  # 标准卷积、空洞率2、空洞率3

        self.conv_weights = nn.Parameter(torch.ones(3))
        self.softmax = nn.Softmax(dim=0)

        self.diff_attention = MDP(
            in_channels=dim,
            max_levels=max_diff_levels
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        x_orig = x

        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # 改进4：动态生成不同空洞率的卷积结果
        conv_outs = []
        norm_weights = self.softmax(self.conv_weights)
        for i, dilation in enumerate(self.dilations):
            if dilation == 1:
                conv_out = self.base_conv(x)
            else:
                conv_out = F.conv2d(
                    x,
                    self.base_conv.weight,
                    bias=self.base_conv.bias,
                    padding=dilation,
                    dilation=dilation,
                    groups=self.dim
                )
            conv_outs.append(conv_out * norm_weights[i])
        #(f"[ViTSubBlock] conv_outs: {[o.shape for o in conv_outs]}")
        x = self.diff_attention(*conv_outs)

        x = x.flatten(2).transpose(1, 2)
        x = x_orig.flatten(2).transpose(1, 2) + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm1(x)))
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)
    
          
