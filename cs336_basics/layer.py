import torch
from torch import nn, Tensor
from torch.nn import Parameter, init
from einops import rearrange, einsum
from math import sqrt
from jaxtyping import Float, Int


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        std = (2 / (in_features + out_features)) ** 0.5
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: Tensor):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embeddings = Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        init.trunc_normal_(self.embeddings, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.embeddings[token_ids] 
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.g = Parameter(torch.ones(d_model, dtype=dtype, device=device))

    def forward(self, x: Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        root_mean_square_x = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        result = (x / root_mean_square_x) * self.g 
        return result.to(in_dtype)
    

def silu(x: Tensor):
    return x * torch.sigmoid(x);


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor):
        return self.linear2( silu(self.linear1(x)) * self.linear3(x) )
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE.")
        self.d_k = d_k

        freq = 1 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        positions = torch.arange(max_seq_len, device=device).float()
        freqs = torch.outer(positions, freq)

        self.register_buffer('cos_cached', torch.cos(freqs),persistent=False)
        self.register_buffer('sin_cached', torch.sin(freqs), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        if x.size(-1) != self.d_k:
            raise ValueError(f"Last dim of x ({x.size(-1)}) ≠ d_k ({self.d_k}).")
        
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        self.cos_pos = self.cos_cached[token_positions]
        self.sin_pos = self.sin_cached[token_positions]

        out_even = x_even * self.cos_pos - x_odd * self.sin_pos
        out_odd = x_even * self.sin_pos + x_odd * self.cos_pos

        out = torch.empty_like(x)
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd

        return out


def softmax(x: Tensor, dim=int):
    out = torch.exp(x - x.max(dim=dim, keepdim=True).values)
    return out / out.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None):
    attn_scores = einsum(q, k, "... q d_k, ... k d_k -> ... q k") / sqrt(q.shape[-1])
    
    if mask is not None:
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

    attn_probs = softmax(attn_scores, dim=-1)
    return einsum(attn_probs, v, "... q k, ... k d_v -> ... q d_v")


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        use_rope: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.d_k = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype) 
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device))
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)

    def forward(
        self,
        x: Float[Tensor, "batch seq_len d_model"],
        token_positions: Int[Tensor, "batch seq_len"] | None = None,
    ) -> Float[Tensor, "batch seq_len d_model"]:
        S = x.shape[-2]

        Q, K, V = [ rearrange(proj(x), "b s (h d) -> b h s d", h=self.num_heads) 
                 for proj in [self.q_proj, self.k_proj, self.v_proj]]
        
        if self.use_rope: Q, K = self.rope(Q, token_positions), self.rope(K, token_positions)

        out = scaled_dot_product_attention(Q, K, V, mask=self.causal_mask[..., :S, :S])

        out = rearrange(out, "b h s d -> b s (h d)")
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float = 10_000.0,
        use_rope: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)

        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len, rope_theta, use_rope, device=device, dtype=dtype)

        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.attn(self.norm1(x), token_positions) + x
        return out + self.ff(self.norm2(out))
    

class TransformerLM(nn.Module):
    def __init__(self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, max_seq_len=context_length, rope_theta=rope_theta, use_rope=True, device=device, dtype=dtype) for _ in range(num_layers)])

        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.final_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.context_length = context_length

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, s = token_ids.shape
        if s > self.context_length:
            raise ValueError(f"seq_len {s} exceeds context_length {self.context_length}")
        
        embeddings = self.embeddings(token_ids)

        positions = torch.arange(s, device=token_ids.device).unsqueeze(0).expand(b, s)

        out = embeddings
        for block in self.transformer_blocks:
            out = block(out, positions)

        return self.final_linear(self.final_norm(out))