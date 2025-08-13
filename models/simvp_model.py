import torch
from torch import nn

from openstl.modules import (ConvSC, ViTSubBlock)


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch.nn.init import trunc_normal_

# ---------- 1. HybridDecompConv：ConvSC 替代 ----------
class HybridDecompConv(nn.Module):
    def __init__(self, C_in, C_out, n_bands=4):
        super().__init__()
        self.decomp = HybridDecomp(C_in, n_bands=n_bands)
        self.proj = nn.Linear(C_in * n_bands, C_out)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # [B, HW, C]
        bands = self.decomp(x)  # list of [B, HW, C]
        x_fused = torch.cat(bands, dim=-1)  # [B, HW, C*n_bands]
        x_proj = self.proj(x_fused)  # [B, HW, C_out]
        x_proj = x_proj.transpose(1, 2).view(B, -1, H, W)  # [B, C_out, H, W]
        return x_proj

# ---------- 2. HybridDecomp ----------
class HybridDecomp(nn.Module):
    def __init__(self, d_v: int, n_bands=4):
        super().__init__()
        self.d_v = d_v
        self.n_bands = n_bands
        self.conv_decomp = MultiScaleConvDecomp(d_v, kernel_sizes=[3, 5, 7, 11][:n_bands])
        self.freq_decomp = FreqDecomp(d_v, n_bands)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> list:
        conv_bands = self.conv_decomp(x)
        target_lengths = [b.size(1) for b in conv_bands]
        freq_bands = self.freq_decomp(x, target_lengths=target_lengths)

        alpha = torch.sigmoid(self.alpha)
        fused = [alpha * c + (1 - alpha) * f for c, f in zip(conv_bands, freq_bands)]
        return fused

# ---------- 3. FreqDecomp ----------
class FreqDecomp(nn.Module):
    def __init__(self, d_v: int, n_bands: int = 4):
        super().__init__()
        self.d_v = d_v
        self.n_bands = n_bands

    def forward(self, x: torch.Tensor, target_lengths: list = None) -> list:
        seq_len = x.size(1)
        x_freq = torch.fft.rfft(x, dim=1)
        freqs = torch.fft.rfftfreq(seq_len, device=x.device)

        n_freqs = len(freqs)
        band_size = n_freqs // self.n_bands

        x_bands = []
        for i in range(self.n_bands):
            mask = torch.zeros_like(freqs, dtype=torch.bool)
            start = i * band_size
            end = (i + 1) * band_size if i < self.n_bands - 1 else n_freqs
            mask[start:end] = True
            band_freq = x_freq * mask.unsqueeze(-1).to(x_freq.dtype)
            band_time = torch.fft.irfft(band_freq, n=seq_len, dim=1)

            if target_lengths is not None:
                length = target_lengths[i]
                band_time_resized = F.interpolate(
                    band_time.transpose(1, 2),
                    size=length,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            else:
                band_time_resized = F.avg_pool1d(
                    band_time.transpose(1, 2), kernel_size=2, stride=2
                ).transpose(1, 2)

            x_bands.append(band_time_resized)
        return x_bands

# ---------- 4. MultiScaleConvDecomp ----------
class MultiScaleConvDecomp(nn.Module):
    def __init__(self, d_v: int, kernel_sizes: List[int]):
        super().__init__()
        self.n_bands = len(kernel_sizes)
        self.filters = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_v, d_v, kernel_size=k, padding=k // 2, groups=d_v, bias=False),
                nn.BatchNorm1d(d_v),
                nn.GELU()
            ) for k in kernel_sizes
        ])
        self.scale_weights = nn.Parameter(torch.ones(self.n_bands))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = x.transpose(1, 2)  # [B, d_v, T]
        norm_weights = torch.softmax(self.scale_weights, dim=0)
        bands = [filt(x).transpose(1, 2) * w for filt, w in zip(self.filters, norm_weights)]
        return bands

# ---------- 5. 编码器 ----------
class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            HybridDecompConv(C_in, C_hid),
            *[HybridDecompConv(C_hid, C_hid) for _ in samplings[1:]]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1

# ---------- 6. 解码器 ----------
class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[HybridDecompConv(C_hid, C_hid) for _ in range(N_S)]
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y

# ---------- 7. SimVP 主模型 ----------
class SimVP_Model(nn.Module):
    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))
        act_inplace = False
        C = 1

        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace)

        model_type = 'gsta' if model_type is None else model_type.lower()
    
        self.hid = MidMetaNet(T * hid_S, hid_T, N_T,
                                  input_resolution=(H, W), model_type=model_type,
                                  mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

    def forward(self, x_raw, **kwargs):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B * T, C, H, W)
        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.view(B * T, C_, H_, W_)
        Y = self.dec(hid, skip)
        Y = Y.view(B, T, C, H, W)
        return Y

# ---------- 8. Sampling 控制函数 ----------
def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    return list(reversed(samplings[:N])) if reverse else samplings[:N]




class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'


        if model_type == 'vit':
            self.block = ViTSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        else:
            assert False and "Invalid model_type in SimVP"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 1 and mlp_ratio > 1
        self.N2 = N2
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, max(1, self.N2))]

        enc_layers = []
        print("N2=", N2)
        for i in range(N2):
            enc_layers.append(
                MetaBlock(
                    channel_in, channel_in,  # 输入输出通道一致
                    input_resolution, model_type,
                    mlp_ratio, drop,
                    drop_path=dpr[i], layer_i=i
                )
            )

        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
        y = z.reshape(B, T, C, H, W)
        return y
