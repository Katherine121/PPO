import os

import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat


def get_pad_mask(seq, pad_idx):
    """
    get the padding mask for indicating valid frames in the input sequence
    :param seq: shape of (b, len)
    :param pad_idx: 0
    :return: shape of (b, 1, len), if not equals to 0, set to 1, if equals to 0, set to 0
    """
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    """
    get the subsequent mask for masking the future frames
    :param seq: shape of (b, len)
    :return: lower triangle shape of (b, len, len)
    """
    b, s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, s, s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class FeatureEnhanceModule(nn.Module):
    def __init__(self, backbone, pretrained):
        """
        Feature Enhance Module
        :param backbone: backbone of Feature Enhance Module (MobileNetV3 small)
        """
        super(FeatureEnhanceModule, self).__init__()
        self.backbone = backbone
        # for name, param in self.backbone.named_parameters():
        #     print(param)

    def forward(self, x):
        """
        forward pass of Feature Enhance Module
        :param x: the provided input tensor
        :return: the visual semantic features of input
        """
        x = self.backbone(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        """
        pre normalization
        :param dim: input dimension of the last axis
        :param fn: next module
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, mask, **kwargs):
        """
        forward pass of PreNorm
        :param x: the provided input tensor
        :return: the visual semantic features of input
        """
        return self.fn(self.norm(x), mask, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """
        full connection layer.
        :param dim: input dimension.
        :param hidden_dim: hidden dimension.
        :param dropout: dropout rate.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask):
        """
        forward pass of FeedForward
        :param x: the provided input tensor
        :return: the visual semantic features of input
        """
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        """
        masked multi-head self attention.
        :param dim: input dimension.
        :param heads: the number of heads.
        :param dim_head: dimension of one head.
        :param dropout: dropout rate.
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask):
        """
        forward pass of Attention
        :param x: the provided input tensor
        :param mask: padding and subsequent mask
        :return: the visual semantic features of input
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            # for head axis broadcasting
            mask = mask.unsqueeze(1)
            dots = dots.masked_fill(mask == 0, -1e9)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttentionModule(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        """
        Cross Attention Module.
        :param dim: input dimension.
        :param depth: depth of Cross Attention Module (Transformer Decoder).
        :param heads: the number of heads in Masked MSA.
        :param dim_head: dimension of one head.
        :param mlp_dim: hidden dimension in FeedForward.
        :param dropout: dropout rate.
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None):
        """
        forward pass of Cross Attention Module
        :param x: the provided input tensor
        :param mask: padding and subsequent mask
        :return: the visual semantic features of input
        """
        for attn, ff in self.layers:
            x = attn(x, mask) + x
            x = ff(x, mask) + x
        return x


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = torch.zeros(shape).cuda()
        self.S = torch.zeros(shape).cuda()
        self.std = torch.sqrt(self.S)

    def update(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.clone()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class ARTransformer(nn.Module):
    def __init__(self, *, backbone, backbone_pretrained=None, extractor_dim,
                 action_dim=2, len=6,
                 dim=512, depth=4, heads=8, dim_head=64, mlp_dim=1024,
                 dropout=0.1, emb_dropout=0.1, is_actor=True):
        """
        ARTransformer
        :param backbone: backbone of Feature Enhance Module (MobileNetV3 small)
        :param extractor_dim: output dimension of Feature Enhance Module
        :param num_classes1: output dimension of ARTransformer
        :param num_classes2: output dimension of ARTransformer
        :param len: input sequence length of ARTransformer
        :param dim: input dimension of Cross Attention Module
        :param depth: depth of Cross Attention Module
        :param heads: the number of heads in Multi-Head Self Attention layer
        :param dim_head: dimension of one head
        :param mlp_dim: hidden dimension in FeedForward layer
        :param dropout: dropout rate
        :param emb_dropout: dropout rate after position embedding
        """
        super().__init__()
        self.extractor = backbone
        self.extractor_dim = 2
        self.norm = Normalization(shape=(2*self.extractor_dim,))

        self.head_angle = nn.Sequential(
            nn.Linear(2 * self.extractor_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, img):
        """
        forward pass of ARTransformer
        :param img: input frame sequence
        :param ang: input angle sequence
        :return: the current position preds, the next position preds, the direction angle preds
        """
        if len(img.shape) == 4:
            img = self.extractor(img)
            img = img.view(-1, 2 * self.extractor_dim)
            pos = img

        else:
            b = img.size(0)
            # b,len,3,224,224->b*len,3,224,224->b*len,576->b,len,576
            img = self.extractor(img.view(b * 2, 3, 224, 224))
            img = img.view(b, 2 * self.extractor_dim)
            pos = img

        # img /= 100
        # mean = torch.mean(img, dim=-1, keepdim=True)
        # std = torch.std(img, dim=-1, keepdim=True)
        # img = (img - mean) / std

        # mean = torch.mean(img, dim=-1, keepdim=True)
        # std = torch.std(img, dim=-1, keepdim=True)
        # img = (img - mean) / std
        img = self.norm(img)

        # b,dim->b,2*dim->2,b,dim
        return self.head_angle(img)


class Critic(nn.Module):
    def __init__(self, *, backbone, backbone_pretrained=None, extractor_dim,
                 action_dim=1, len=6,
                 dim=512, depth=4, heads=8, dim_head=64, mlp_dim=1024,
                 dropout=0.1, emb_dropout=0.1, is_actor=True):
        """
        ARTransformer
        :param backbone: backbone of Feature Enhance Module (MobileNetV3 small)
        :param extractor_dim: output dimension of Feature Enhance Module
        :param num_classes1: output dimension of ARTransformer
        :param num_classes2: output dimension of ARTransformer
        :param len: input sequence length of ARTransformer
        :param dim: input dimension of Cross Attention Module
        :param depth: depth of Cross Attention Module
        :param heads: the number of heads in Multi-Head Self Attention layer
        :param dim_head: dimension of one head
        :param mlp_dim: hidden dimension in FeedForward layer
        :param dropout: dropout rate
        :param emb_dropout: dropout rate after position embedding
        """
        super().__init__()
        self.extractor1 = backbone
        self.extractor_dim1 = 2
        self.norm = Normalization(shape=(2*self.extractor_dim1,))

        self.head_angle1 = nn.Sequential(
            nn.Linear(2 * self.extractor_dim1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, img):
        """
        forward pass of ARTransformer
        :param img: input frame sequence
        :param ang: input angle sequence
        :return: the current position preds, the next position preds, the direction angle preds
        """
        if len(img.shape) == 4:
            img = self.extractor1(img)
            img = img.view(-1, 2 * self.extractor_dim1)
            pos = img
        else:
            b = img.size(0)
            # b,len,3,224,224->b*len,3,224,224->b*len,576->b,len,576
            img = self.extractor1(img.view(b * 2, 3, 224, 224))
            img = img.view(b, 2 * self.extractor_dim1)
            pos = img

        # img /= 100
        # mean = torch.mean(img, dim=-1, keepdim=True)
        # std = torch.std(img, dim=-1, keepdim=True)
        # img = (img - mean) / std

        # mean = torch.mean(img, dim=-1, keepdim=True)
        # std = torch.std(img, dim=-1, keepdim=True)
        # img = (img - mean) / std
        img = self.norm(img)

        # b,dim->b,2*dim->2,b,dim
        return self.head_angle1(img)
