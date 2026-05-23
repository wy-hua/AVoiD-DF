"""
Multi-Modal Joint-Decoder components (MMD), as described in:
  AVoiD-DF: Audio-Visual Joint Learning for Detecting Deepfake
  Equations 9-14, Figure 4.
"""
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Standard ViT 2D patch embedding for a single modality."""

    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input size ({H}x{W}) doesn't match expected ({self.img_size[0]}x{self.img_size[1]})"
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, N, embed_dim]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x)), attn


class CrossAttention(nn.Module):
    """One direction of BiCroAtt: Q from x, K/V from y (Eq. 9-10)."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, Nx, C = x.shape
        Ny = y.shape[1]
        H = self.num_heads
        D = C // H
        q = self.wq(x).reshape(B, Nx, H, D).permute(0, 2, 1, 3)
        k = self.wk(y).reshape(B, Ny, H, D).permute(0, 2, 1, 3)
        v = self.wv(y).reshape(B, Ny, H, D).permute(0, 2, 1, 3)
        attn = self.attn_drop((q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1))
        out = self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, Nx, C)))
        return out, attn


class Encoder_layer(nn.Module):
    """Standard transformer encoder block: self-attention + FFN (Eq. 8)."""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., norm_layer=nn.LayerNorm, act_layer=None):
        super().__init__()
        act_layer = act_layer or nn.GELU
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, attn_drop=attn_drop_ratio,
                                  proj_drop=drop_ratio)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x))
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class MMD(nn.Module):
    """
    One layer of the Multi-Modal Joint-Decoder (Eq. 12-14, Figure 4).

    Architecture per layer:
      BiCroAtt (bidirectional cross-attention) → SelfAtt → FF  for each stream.
    Returns updated (video, audio) features and their cross-attention weight maps.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_ratio=0., attn_drop_ratio=0., norm_layer=nn.LayerNorm, act_layer=None):
        super().__init__()
        act_layer = act_layer or nn.GELU

        # BiCroAtt: video Q → audio K/V and audio Q → video K/V
        self.cross_v2a = CrossAttention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop_ratio, proj_drop=drop_ratio)
        self.cross_a2v = CrossAttention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop_ratio, proj_drop=drop_ratio)

        # SelfAtt for each stream (Eq. 13)
        self.self_attn_v = SelfAttention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop_ratio, proj_drop=drop_ratio)
        self.self_attn_a = SelfAttention(dim, num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         attn_drop=attn_drop_ratio, proj_drop=drop_ratio)

        # Layer norms: 3 per stream (before BiCroAtt, SelfAtt, FF)
        self.norm_v1 = norm_layer(dim)
        self.norm_a1 = norm_layer(dim)
        self.norm_v2 = norm_layer(dim)
        self.norm_a2 = norm_layer(dim)
        self.norm_v3 = norm_layer(dim)
        self.norm_a3 = norm_layer(dim)

        # FF (Eq. 14)
        mlp_hidden = int(dim * mlp_ratio)
        self.ff_v = Mlp(in_features=dim, hidden_features=mlp_hidden, act_layer=act_layer, drop=drop_ratio)
        self.ff_a = Mlp(in_features=dim, hidden_features=mlp_hidden, act_layer=act_layer, drop=drop_ratio)

    def forward(self, inputs):
        x, y, enc_v, enc_a = inputs  # x=video, y=audio

        # Eq. 12: BiCroAtt with residual
        cross_v, attn_v = self.cross_v2a(self.norm_v1(x), self.norm_a1(y))
        cross_a, attn_a = self.cross_a2v(self.norm_a1(y), self.norm_v1(x))
        x = x + cross_v
        y = y + cross_a

        # Eq. 13: SelfAtt with residual
        self_v, _ = self.self_attn_v(self.norm_v2(x))
        self_a, _ = self.self_attn_a(self.norm_a2(y))
        x = x + self_v
        y = y + self_a

        # Eq. 14: FF with residual
        x = x + self.ff_v(self.norm_v3(x))
        y = y + self.ff_a(self.norm_a3(y))

        return x, y, attn_v, attn_a
