# modified code from https://github.com/megvii-research/FQ-ViT/tree/main
# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import collections.abc
import math
import os
import re
import warnings
from collections import OrderedDict
from functools import partial
from itertools import repeat
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .base_model import Base_Model


__all__ = [
    'DeiT_Tiny_Patch16_224', 'DeiT_Small_Patch16_224', 'DeiT_Base_Patch16_224'
]


@torch.no_grad()
def load_weights_from_npz(model,
                          url,
                          check_hash=False,
                          progress=False,
                          prefix=''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    def _get_cache_dir(child_dir=''):
        """
        Returns the location of the directory where models are cached (and creates it if necessary).
        """
        hub_dir = torch.hub.get_dir()
        child_dir = () if not child_dir else (child_dir,)
        model_dir = os.path.join(hub_dir, 'checkpoints', *child_dir)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def _download_cached_file(url, check_hash=True, progress=False):
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(_get_cache_dir(), filename)
        if not os.path.exists(cached_file):
            hash_prefix = None
            if check_hash:
                r = torch.hub.HASH_REGEX.search(
                    filename)  # r is Optional[Match[str]]
                hash_prefix = r.group(1) if r else None
            torch.hub.download_url_to_file(url,
                                           cached_file,
                                           hash_prefix,
                                           progress=progress)
        return cached_file

    def adapt_input_conv(in_chans, conv_weight):
        conv_type = conv_weight.dtype
        # Some weights are in torch.half, ensure it's float for sum on CPU
        conv_weight = conv_weight.float()
        O, I, J, K = conv_weight.shape
        if in_chans == 1:
            if I > 3:
                assert conv_weight.shape[1] % 3 == 0
                # For models with space2depth stems
                conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
                conv_weight = conv_weight.sum(dim=2, keepdim=False)
            else:
                conv_weight = conv_weight.sum(dim=1, keepdim=True)
        elif in_chans != 3:
            if I != 3:
                raise NotImplementedError(
                    'Weight format not supported by conversion.')
            else:
                # NOTE this strategy should be better than random init, but there could be other combinations of
                # the original RGB input layer weights that'd work better for specific cases.
                repeat = int(math.ceil(in_chans / 3))
                conv_weight = conv_weight.repeat(1, repeat, 1,
                                                 1)[:, :in_chans, :, :]
                conv_weight *= (3 / float(in_chans))
        conv_weight = conv_weight.to(conv_type)
        return conv_weight

    def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
        # Rescale the grid of position embeddings when loading from state_dict. Adapted from
        # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
        ntok_new = posemb_new.shape[1]
        if num_tokens:
            posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[
                                                              0, num_tokens:]
            ntok_new -= num_tokens
        else:
            posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
        gs_old = int(math.sqrt(len(posemb_grid)))
        if not len(gs_new):  # backwards compatibility
            gs_new = [int(math.sqrt(ntok_new))] * 2
        assert len(gs_new) >= 2
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old,
                                          -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid,
                                    size=gs_new,
                                    mode='bicubic',
                                    align_corners=False)
        posemb_grid = posemb_grid.permute(0, 2, 3,
                                          1).reshape(1, gs_new[0] * gs_new[1],
                                                     -1)
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        return posemb

    cached_file = _download_cached_file(url,
                                        check_hash=check_hash,
                                        progress=progress)

    w = np.load(cached_file)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(
            adapt_input_conv(stem.conv.weight.shape[1],
                             _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(
                            _n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(
                            _n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(
                            _n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(
                            _n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(
                            _n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(
                            _n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(model.patch_embed.proj.weight.shape[1],
                                        _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'],
                       t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1),
            model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(
            model.head, nn.Linear
    ) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None),
                  nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(
            torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T
                for n in ('query', 'key', 'value')
            ]))
        block.attn.qkv.bias.copy_(
            torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1)
                for n in ('query', 'key', 'value')
            ]))
        block.attn.proj.weight.copy_(
            _n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(
                _n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(
                _n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + \
                    torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        if norm_layer:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
                H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if isinstance(self.norm, nn.Identity):
            x = self.norm(x)
        else:
            x = self.norm(x, self.qact_before_norm.quantizer,
                          self.qact.quantizer)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        x = self.qkv(x)
        qkv = x.reshape(B, N, 3, self.num_heads,
                        C // self.num_heads).permute(2, 0, 3, 1, 4)  # (BN33)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 hybrid_backbone=None,
                 norm_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict([
                    ('fc', nn.Linear(embed_dim, representation_size)),
                    ('act', nn.Tanh()),
                ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = (nn.Linear(self.num_features, num_classes, bias=True)
                     if num_classes > 0 else nn.Identity())
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = (nn.Linear(self.embed_dim, num_classes)
                     if num_classes > 0 else nn.Identity())

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@Base_Model.register('deit_tiny')
class DeiT_Tiny_Patch16_224(Base_Model, VisionTransformer):
    def __init__(self, num_classes=1000, pretrained=False, progress=True,
                 **kwargs):
        VisionTransformer.__init__(self, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
                                   qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
        Base_Model.__init__(self, **kwargs)
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
                self.save_path,
                map_location='cpu',
                check_hash=True,
                progress=progress
            )
            self.load_state_dict(checkpoint['model'], strict=False)

@Base_Model.register('deit_small')
class DeiT_Small_Patch16_224(Base_Model, VisionTransformer):
    def __init__(self, num_classes=1000, pretrained=False, progress=True,
                 **kwargs):
        VisionTransformer.__init__(self, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
                                   qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
        Base_Model.__init__(self, **kwargs)
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
                self.save_path,
                map_location='cpu',
                check_hash=True,
                progress=progress
            )
            self.load_state_dict(checkpoint['model'], strict=False)


@Base_Model.register('deit_base')
class DeiT_Base_Patch16_224(Base_Model, VisionTransformer):
    def __init__(self, num_classes=1000, pretrained=False, progress=True,
                 **kwargs):
        VisionTransformer.__init__(self, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                   qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes)
        Base_Model.__init__(self, **kwargs)
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
                self.save_path,
                map_location='cpu',
                check_hash=True,
                progress=progress
            )
            self.load_state_dict(checkpoint['model'], strict=False)

