import torch
from torch import nn


class SCA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = q.shape

        q = self.q(q).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads, H * W, -1)
        k = self.k(k).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads, H * W, -1)
        v = self.v(v).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3).reshape(B * self.num_heads, H * W, -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class Adapter(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 0.25):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc_down = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc_up = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_up(self.act(self.fc_down(x)))
