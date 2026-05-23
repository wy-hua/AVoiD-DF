import copy
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from model.MMD import MMD, PatchEmbed, Encoder_layer


class AVoiD(nn.Module):
    def __init__(self, args, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=5, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, device='cuda:0'):

        super(AVoiD, self).__init__()
        self.dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_ratio = drop_ratio
        self.attn_drop_ratio = attn_drop_ratio
        self.act_layer = act_layer
        self.depth = depth
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed_video = embed_layer(img_size=img_size, patch_size=patch_size,
                                             in_c=in_c, embed_dim=embed_dim)
        self.patch_embed_audio = embed_layer(img_size=img_size, patch_size=patch_size,
                                             in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed_video.num_patches

        self.cls_token_video = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_audio = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None

        self.pos_embed_video = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_embed_audio = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop_video = nn.Dropout(p=drop_ratio)
        self.pos_drop_audio = nn.Dropout(p=drop_ratio)

        self.time_embed_video = nn.Parameter(torch.zeros(1, embed_dim))
        self.time_embed_audio = nn.Parameter(torch.zeros(1, embed_dim))
        self.time_drop_video = nn.Dropout(p=drop_ratio)
        self.time_drop_audio = nn.Dropout(p=drop_ratio)

        # MMD blocks (depth-1 layers)
        self.block = nn.ModuleList([
            MMD(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                act_layer=act_layer)
            for _ in range(depth - 1)
        ])

        # Final encoder layer applied to selected patches
        self.last_block = Encoder_layer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio)

        # Unimodal spatial encoders (E_spa)
        self.video_encoder = nn.Sequential(*[
            Encoder_layer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio)
            for _ in range(6)
        ])
        self.audio_encoder = nn.Sequential(*[
            Encoder_layer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio)
            for _ in range(6)
        ])

        self.norm = norm_layer(embed_dim)
        # Fuse video + audio patch sequences → shared space
        self.av_fc = nn.Linear(embed_dim * 2, embed_dim)
        # Combine cls_v + fusion_cls + cls_a → final feature
        self.fc = nn.Linear(embed_dim * 3, embed_dim)
        # Classification head
        self.has_logits = False
        self.head_dist = None
        self.head = nn.Linear(embed_dim, num_classes)

        # Learnable fusion weights
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))
        self.w3 = nn.Parameter(torch.ones(1))

        # Weight init
        nn.init.trunc_normal_(self.pos_embed_video, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_audio, std=0.02)
        nn.init.trunc_normal_(self.cls_token_video, std=0.02)
        nn.init.trunc_normal_(self.cls_token_audio, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        self.apply(_init_vit_weights)
        self.device = device

    def forward_features(self, video, audio):
        x = self.patch_embed_video(video)   # [B, N, D]
        y = self.patch_embed_audio(audio)   # [B, N, D]

        cls_token_video = self.cls_token_video.expand(x.shape[0], -1, -1)
        cls_token_audio = self.cls_token_audio.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token_video, x), dim=1)   # [B, N+1, D]
        y = torch.cat((cls_token_audio, y), dim=1)

        # Position + temporal embedding
        x = self.pos_drop_video(x + self.pos_embed_video)
        y = self.pos_drop_audio(y + self.pos_embed_audio)
        x = self.time_drop_video(x + self.time_embed_video)
        y = self.time_drop_audio(y + self.time_embed_audio)  # fixed: was time_embed_video

        # Unimodal spatial encoders
        x = self.video_encoder(x)   # [B, N+1, D]
        y = self.audio_encoder(y)

        Encoder_video = x
        Encoder_audio = y

        # CLS tokens before fusion
        cls_v = x[:, 0, :]   # [B, D]
        cls_a = y[:, 0, :]

        # MMD blocks — bidirectional cross-attention
        weight_list_v, weight_list_a = [], []
        for b in self.block:
            x, y, w_v, w_a = b((x, y, Encoder_video, Encoder_audio))
            weight_list_v.append(w_v)   # [B, heads, N+1, N+1]
            weight_list_a.append(w_a)

        # Fuse video + audio
        xy = self.av_fc(torch.cat((x, y), dim=-1))   # [B, N+1, D]

        # Part selection: top-k patches by mean cross-attention to cls token
        attn_v = torch.stack(weight_list_v, dim=0).mean(0)   # [B, heads, N+1, N+1]
        attn_a = torch.stack(weight_list_a, dim=0).mean(0)
        cls_attn = ((attn_v[:, :, 0, 1:] + attn_a[:, :, 0, 1:]) / 2).mean(1)  # [B, N]
        k = min(4, cls_attn.shape[1])
        top_idx = cls_attn.topk(k, dim=1)[1] + 1   # +1 to skip cls token position

        B = x.shape[0]
        parts = torch.stack([xy[i, top_idx[i]] for i in range(B)], dim=0)   # [B, k, D]
        concat_va = torch.cat((xy[:, 0:1], parts), dim=1)   # [B, k+1, D]

        x = self.last_block(concat_va)   # [B, k+1, D]
        fusion_cls = x[:, 0]             # [B, D]

        last_cls = self.fc(torch.cat(
            (self.w1 * cls_v, self.w2 * fusion_cls, self.w3 * cls_a), dim=-1
        ))
        return last_cls, cls_v, cls_a

    def forward(self, video, audio):
        feat, cls_v, cls_a = self.forward_features(video, audio)
        out = self.head(feat)
        return out, feat, cls_v, cls_a


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def AVoiD_mm(args, num_classes: int = 2, has_logits: bool = False):
    model = AVoiD(args=args,
                  img_size=224,
                  patch_size=16,
                  embed_dim=768,
                  depth=6,
                  num_heads=12,
                  representation_size=768 if has_logits else None,
                  num_classes=num_classes,
                  device=args.device)
    return model
