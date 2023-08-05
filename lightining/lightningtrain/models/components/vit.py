from typing import Any, Optional
import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce
from torch.nn import functional as F

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels=3,
        patch_size=16,
        emb_size=768,
        img_size=224,
    ):
        super(PatchEmbedding, self).__init__()

        assert (
            img_size / patch_size % 1 == 0
        ), "img_size must be integer multiple of patch_size"

        self.projection = nn.Sequential(
            Rearrange(
                "b c (h s1) (w s2) -> b (h w) (s1 s2 c)", s1=patch_size, s2=patch_size
            ),
            nn.Linear(patch_size * patch_size * in_channels, emb_size),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

        self.positional_emb = nn.Parameter(
            torch.randn(
                (img_size // patch_size) * (img_size // patch_size)
                + 1,  
                emb_size,
            )
        )

    def forward(self, x: torch.Tensor):
        B, c, h, w = x.shape
        x = self.projection(x)
        b=B
        cls_token = self.cls_token.repeat(b,1,1)

        # print(cls_token.shape)

        x = torch.cat([cls_token, x], dim=1)

        x += self.positional_emb

        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, dropout=0):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.emb_size = emb_size

        self.query = nn.Linear(emb_size, emb_size)
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)

        self.projection = nn.Linear(emb_size, emb_size)

        self.attn_dropout = nn.Dropout(dropout)

        self.scaling = (self.emb_size // num_heads) ** -0.5

        self.rearrange_heads = Rearrange(
            "batch seq_len (num_head h_dim) -> batch num_head seq_len h_dim", num_head=self.num_heads
        )
        self.rearrange_out = Rearrange("batch num_head seq_length dim -> batch seq_length (num_head dim)")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
    
        queries = self.rearrange_heads(self.query(x))
        keys = self.rearrange_heads(self.key(x))
        values = self.rearrange_heads(self.value(x))


        energies = torch.einsum("bhqd, bhkd -> bhqk", queries, keys)

        if mask is not None:
            fill_value = torch.finfo(energies.dtype).min
            energies.mask_fill(~mask, fill_value)

        attention = F.softmax(energies, dim=-1) * self.scaling

        attention = self.attn_dropout(attention)

        out = torch.einsum("bhas, bhsd -> bhad", attention, values)

        out = self.rearrange_out(out)
        out = self.projection(out)

        return out
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()

        self.fn = fn

    def forward(self, x):
        res = x

        out = self.fn(x)

        out += res

        return out

FeedForwardBlock = lambda emb_size=768, expansion=4, drop_p=0.0: nn.Sequential(
    nn.Linear(emb_size, expansion * emb_size),
    nn.GELU(),
    nn.Dropout(drop_p),
    nn.Linear(expansion * emb_size, emb_size),
)

class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self, emb_size=768, drop_p=0.0, forward_expansion=4, forward_drop_p=0
    ):
        super(TransformerEncoderBlock, self).__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=12, emb_size=768):
        super(TransformerEncoder, self).__init__()

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=768, num_classes=1000):
        super(ClassificationHead, self).__init__(
            Reduce(
                "batch_size seq_len emb_dim -> batch_size emb_dim", reduction="mean"
            ),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes),
        )

class ViT(nn.Sequential):
    def __init__(
        self,
        in_channels=3,
        patch_size=16,
        emb_size=768,
        img_size=224,
        depth=12,
        num_classes=1000
    ):
        super(ViT, self).__init__(
            PatchEmbedding(
                in_channels,
                patch_size,
                emb_size,
                img_size,
            ),
            TransformerEncoder(depth, emb_size=emb_size),
            ClassificationHead(emb_size, num_classes),
        )

class ToTensorModule(nn.Module):
    def forward(self, x):
        return torch.tensor(x)
