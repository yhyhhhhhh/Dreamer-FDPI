import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentLayer(nn.Module):
    def __init__(self, inp_size, size, hidden, act):
        super().__init__()
        self.layer = nn.Sequential(
            FeedForwardLayer(inp_size, hidden, act()),
            FeedForwardLayer(hidden, hidden, act()),
        )
        self.head = nn.Linear(hidden, size)
        self.head.apply(uniform_weight_init(0.0))

    def forward(self, inp):
        return self.head(self.layer(inp))


class Encoder(nn.Module):
    def __init__(self, width, in_ch, stem_ch, min_res, act):
        super().__init__()
        feature_width = width // 2
        channels = stem_ch

        backbone = [ConvLayer(in_ch, channels, act())]
        while True:
            out_channels = channels * 2
            backbone.append(ConvLayer(channels, out_channels, act()))
            channels = out_channels
            feature_width //= 2
            if feature_width == min_res:
                break

        self.backbone = nn.Sequential(*backbone)
        self.out_ch = channels
        self.embed = self.out_ch * (min_res**2)

    def forward(self, x):
        shape = x.shape[:2]
        x = x.flatten(0, 1)
        x = self.backbone(x)
        x = x.flatten(1, -1)
        return x.unflatten(0, shape)


class Decoder(nn.Module):
    def __init__(self, stoch, out_ch, in_ch, stem_ch, min_res, act):
        super().__init__()
        backbone = [Rearrange(stoch, out_ch, min_res, act())]

        channels = out_ch
        while channels != stem_ch:
            out_channels = channels // 2
            backbone.append(TransposeConvLayer(channels, out_channels, act()))
            channels = out_channels

        backbone.append(nn.ConvTranspose2d(channels, in_ch, 4, 2, 1))
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        shape = sample.shape[:2]
        obs_hat = self.backbone(sample)
        return torch.sigmoid(obs_hat).unflatten(0, shape)


class ProprioEncoder(nn.Module):
    def __init__(self, inp_size, hidden, num_layer, act):
        super().__init__()
        backbone = [FeedForwardLayer(inp_size, hidden, act())]
        for _ in range(num_layer):
            backbone.append(FeedForwardLayer(hidden, hidden, act()))
        self.backbone = nn.Sequential(*backbone)
        self.embed = hidden

    def forward(self, x):
        return self.backbone(x)


class ProprioDecoder(nn.Module):
    def __init__(self, stoch, size, hidden, num_layer, act):
        super().__init__()
        backbone = [FeedForwardLayer(stoch, hidden, act())]
        for _ in range(num_layer):
            backbone.append(FeedForwardLayer(hidden, hidden, act()))
        backbone.append(nn.Linear(hidden, size))
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        return self.backbone(sample)


class Head(nn.Module):
    def __init__(self, inp_size, size, hidden, act):
        super().__init__()
        self.backbone = nn.Sequential(
            FeedForwardLayer(inp_size, hidden, act()),
            FeedForwardLayer(hidden, hidden, act()),
        )
        self.head = nn.Linear(hidden, size)
        self.head.apply(uniform_weight_init(0.0))

    def forward(self, feat):
        return self.head(self.backbone(feat))


class InpLayer(nn.Module):
    def __init__(self, inp_size, size, hidden, act):
        super().__init__()
        self.backbone = FeedForwardLayer(inp_size, hidden, act())
        self.head = nn.Sequential(nn.Linear(hidden, size, bias=False), nn.LayerNorm(size))

    def forward(self, inp):
        return self.head(self.backbone(inp))


class ImsStatLayer(nn.Module):
    def __init__(self, inp_size, size, act):
        super().__init__()
        self.backbone = GatingLayer(inp_size, act())
        self.head = nn.Linear(inp_size, size)
        self.norm = BatchNorm1d(size)

    def forward(self, inp):
        x = self.backbone(inp)
        return self.norm(self.head(x))


class ObsStatLayer(nn.Module):
    def __init__(self, inp_size, size, act):
        super().__init__()
        self.head = nn.Linear(inp_size, size, bias=False)
        self.norm = BatchNorm1d(size)

    def forward(self, inp):
        return self.norm(self.head(inp))


class MixingLayer(nn.Module):
    def __init__(self, inp_size, hidden, bias=True):
        super().__init__()
        self.mix = nn.Parameter(init_weight(1, inp_size))
        self.register_parameter("mix_weight", self.mix)
        self.layer = nn.Linear(inp_size, hidden, bias=bias)

    def forward(self, inp, last):
        mix_w = torch.sigmoid(self.mix + 1)
        return self.layer(mix_w * inp + (1 - mix_w) * last)


class GatingLayer(nn.Module):
    def __init__(self, hidden, act, pdrop=0.1):
        super().__init__()
        self.layer = nn.Linear(hidden, 2 * hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(pdrop)
        self.act = act

    def forward(self, inp):
        x, gate = self.layer(inp).chunk(2, dim=-1)
        x = F.mish(gate) * x
        x = self.drop(x)
        return self.norm(x + inp)


class FeedForwardLayer(nn.Module):
    def __init__(self, inp_size, size, act):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(inp_size, size, bias=False), nn.LayerNorm(size))
        self.act = act

    def forward(self, inp):
        return self.act(self.layer(inp))


class Rearrange(nn.Module):
    def __init__(self, in_dim, out_ch, min_res, act):
        super().__init__()
        out_dim = out_ch * (min_res**2)
        self.layer = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.size = (out_ch, min_res, min_res)

    def forward(self, x):
        x = self.layer(x)
        x = x.flatten(0, 1)
        x = x.unflatten(1, self.size)
        return self.norm(x)


class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 4, 2, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = act

    def forward(self, inp):
        return self.act(self.norm(self.conv(inp)))


class TransposeConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, act):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1, bias=False)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = act

    def forward(self, inp):
        return self.act(self.norm(self.conv(inp)))


class BatchNorm1d(nn.Module):
    def __init__(self, size, eps=1e-5):
        super().__init__()
        self.norm = nn.BatchNorm1d(size, eps=eps)

    def forward(self, inp):
        if inp.dim() > 2:
            shape = inp.shape[:2]
            x = inp.flatten(0, 1)
            x = self.norm(x)
            return x.unflatten(0, shape)
        return self.norm(inp)


class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, inp):
        x = inp.float()
        scale = torch.square(x).mean(dim=-1, keepdim=True)
        scale = torch.rsqrt(scale + self.eps)
        x = (x * scale).type_as(inp)
        return x * self.weight


def init_weight(in_dim, out_dim):
    denoms = (in_dim + out_dim) / 2
    scale = 1 / denoms
    std = math.sqrt(scale) / 0.87962566103423978
    data = torch.randn(in_dim, out_dim) * std
    return torch.clip(data, min=-2 * std, max=2 * std)


def uniform_weight_init(given_scale):
    def init_fn(module):
        if isinstance(module, nn.Linear):
            in_num = module.in_features
            out_num = module.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(module.weight.data, a=-limit, b=limit)
            if hasattr(module.bias, "data"):
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if hasattr(module.bias, "data"):
                module.bias.data.fill_(0.0)

    return init_fn
