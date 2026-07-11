from encodings.quopri_codec import quopri_decode

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 超参数
# =========================

n_embd = 32
num_heads = 4

num_kv_heads = 1 # MQA / GQA

n_layer = 4
block_size = 16
dropout = 0.1

learning_rate = 3e-3
max_iters = 3000
batch_size = 16

device = torch.device(
    "cuda" if torch.cuda.is_available()
    # else "mps" if torch.mps.is_available()
    else "cpu"
)

torch.manual_seed(1337)


# =========================
# 数据
# =========================

text = """
今天天气很好，我想出去走走。
我喜欢学习人工智能，也喜欢研究大模型。
大模型可以根据前面的文字预测后面的文字。
我们现在正在从零开始写一个小语言模型。
"""

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {
    ch: i
    for i, ch in enumerate(chars)
}

itos = {
    i: ch
    for i, ch in enumerate(chars)
}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(
        itos[i]
        for i in ids
    )


data = torch.tensor(
    encode(text),
    dtype=torch.long,
)

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    source = (
        train_data
        if split == "train"
        else val_data
    )

    if len(source) <= block_size:
        raise ValueError(
            f"{split} 数据长度为 {len(source)}，"
            f"必须大于 block_size={block_size}"
        )

    ix = torch.randint(
        len(source) - block_size,
        (batch_size,),
    )

    x = torch.stack([
        source[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        source[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)


# RoPE
class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, block_size, base = 10000.0):
        super().__init__()

        if head_size % 2 != 0:
            raise ValueError("RoPE needs head_size to be even")

        # assume head_size = 8,
        # arange(0, 8, 2) = [0, 2, 4, 6]
        dimension_indices = torch.arange(0, head_size, 2, dtype=torch.float32)
        inv_freq = 1.0 / (
            base ** (dimension_indices / head_size)
        )
        # 1/[base^(2i/d)]
        # [head_size / 2]

        positions = torch.arange(
            block_size,
            dtype=torch.float32,
        )
        # [block_size, ]

        angles = torch.outer(
            positions, inv_freq
        )
        # [block_size, head_size / 2]

        self.register_buffer(
            "cos",
            angles.cos()
        )
        self.register_buffer(
            "sin",
            angles.sin()
        )

    def forward(self, x):

        # x: [B, H, T, head_size]

        B, H, T, D = x.shape

        # even / odd here references to the e/o of indexes in python, which starts from 0 (0 is even)
        # [B, H, T, head_size / 2]
        x_even = x[:, :, :, 0::2]
        x_odd = x[..., 1::2]

        # cos, sin: [block_size, head_size / 2] ----[:T][None, None, :, :]----> [1, 1, T, head_size / 2]
        cos = self.cos[:T][None, None, :, :]
        sin = self.sin[:T][None, None, :, :]

        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        out = torch.stack(
            [
                rotated_even,
                rotated_odd
            ],
            dim = -1
        )
        # After stack: [B, H, T, head_size / 2, 2]

        out = out.flatten(-2)
        # After flattening: [B, H, T, head_size]

        return out


# =========================
# 多头注意力 + fused attn
# =========================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        n_embd,
        num_heads,
        num_kv_heads,
        block_size,
        dropout,
    ):
        super().__init__()

        if n_embd % num_heads != 0:
            raise ValueError(
                "n_embd 必须能被 num_heads 整除"
            )

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                "num_heads must be divisible by num_kv_heads"
            )

        self.n_embd = n_embd
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.head_size = n_embd // num_heads
        self.dropout_p = dropout

        self.num_kv_groups = num_heads // num_kv_heads

        if self.head_size % 2 != 0:
            raise ValueError(
                "Using Rope, needs head_size to be even"
            )

        self.q_dim = num_heads * self.head_size
        self.kv_dim = num_kv_heads * self.head_size

        self.qkv_proj = nn.Linear(
            n_embd,
            self.q_dim + self.kv_dim + self.kv_dim,
            bias = False
        )

        self.rope = RotaryEmbedding(
            head_size = self.head_size,
            block_size = block_size
        )

        self.out_proj = nn.Linear(
            n_embd, n_embd, bias = False
        )

        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B,T,n_embd]

        B, T, C = x.shape

        if T > self.block_size:
            raise ValueError(
                f"series length T = {T} > block_size = {self.block_size}"
            )

        qkv = self.qkv_proj(x)
        # [B, T, q_dim + 2 * kv_dim]

        q, k, v = torch.split(
            qkv,
            [self.q_dim, self.kv_dim, self.kv_dim],
            dim = -1
        )
        # q: [B, T, num_heads * head_size]
        # k: [B, T, num_kv_heads * head_size]
        # v: [B, T, num_kv_heads * head_size]

        q = q.reshape(B, T, self.num_heads, self.head_size)
        k = k.reshape(B, T, self.num_kv_heads, self.head_size)
        v = v.reshape(B, T, self.num_kv_heads, self.head_size)
        # [B, T, H(q or kv), D]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # [B, H(q or kv), T, D]


        q = self.rope(q)
        k = self.rope(k)

        k = k.repeat_interleave(self.num_kv_heads, dim = 1)
        v = v.repeat_interleave(self.num_kv_heads, dim = 1)
        # qkv: [B, H, T, D]

        attention_dropout_p = self.dropout_p if self.training else 0.0

        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=attention_dropout_p,
            is_causal=True
        )
        # [B, H, T, D]

        out = out.transpose(1, 2)
        # [B, T, H, D]

        out = out.contiguous().view(B, T, C)

        out = self.out_proj(out)
        # [B, T, n_embd]

        out = self.out_dropout(out)

        return out


# RMS Norm
class RMSNorm(nn.Module):
    def __init__(self, n_embd, eps = 1e-6):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(n_embd)
        )

    def forward(self, x):
        # x: [B, T, n_embd]
        x_float = x.float()

        # [B, T, 1]
        mean_sq = x_float.pow(2).mean(dim = -1, keepdim = True)
        rms_inv = torch.rsqrt(mean_sq + self.eps)

        # [B, T, n_embd]
        x_norm = x_float * rms_inv

        x_norm = x_norm.to(dtype = x.dtype)

        return x_norm * self.weight


# =========================
# FeedForward / MLP
# =========================

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()

        # 4 times dim
        hidden_dim = 4 * n_embd

        # gate branch
        self.gate_proj = nn.Linear(
            n_embd, hidden_dim, bias=False
        )

        # up
        self.up_proj = nn.Linear(
            n_embd, hidden_dim, bias=False
        )

        # down
        self.down_proj = nn.Linear(
            hidden_dim, n_embd, bias=False
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, n_embd]

        # [B, T, hidden_dim]
        gate = self.gate_proj(x)
        gate = F.silu(gate)

        up = self.up_proj(x)

        hidden = gate * up

        out = self.down_proj(hidden)
        out = self.dropout(out)

        return out


# =========================
# Transformer Block
# =========================

class Block(nn.Module):

    def __init__(
        self,
        n_embd,
        num_heads,
        num_kv_heads,
        block_size,
        dropout,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            n_embd=n_embd,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            dropout=dropout,
        )

        self.rms1 = RMSNorm(n_embd)
        self.rms2 = RMSNorm(n_embd)

        self.feed_forward = FeedForward(
            n_embd=n_embd,
            dropout=dropout,
        )

    def forward(self, x):
        # 第一条残差分支：注意力
        x = x + self.self_attention(
            self.rms1(x)
        )

        # 第二条残差分支：FFN
        x = x + self.feed_forward(
            self.rms2(x)
        )

        return x


# =========================
# 语言模型
# =========================

class MiniGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_embd,
        num_heads,
        num_kv_heads,
        n_layer,
        block_size,
        dropout,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd,
        )

        # Now using RoPE, comment out pe here
        # self.position_embedding_table = nn.Embedding(
        #     block_size,
        #     n_embd,
        # )

        # 暂时只放一个 Transformer Block
        # self.block = Block(
        #     n_embd=n_embd,
        #     num_heads=num_heads,
        #     block_size=block_size,
        #     dropout=dropout,
        # )
        # Multi layers
        self.blocks = nn.ModuleList([
            Block(n_embd = n_embd, num_heads = num_heads, num_kv_heads=num_kv_heads, block_size = block_size, dropout = dropout)
            for _ in range(n_layer)
        ])

        # 所有 Block 后面的最终 LayerNorm
        # Now using RMSNorm
        self.rms_f = RMSNorm(n_embd)

        # hidden state -> 词表 logits
        self.lm_head = nn.Linear(
            n_embd,
            vocab_size,
            bias=False
        )

        self.lm_head.weight = self.token_embedding_table.weight

        nn.init.normal_(
            self.token_embedding_table.weight,
            mean=0.0,
            std=0.02,
        )

    def forward(
        self,
        idx,
        targets=None,
    ):
        B, T = idx.shape

        # tok_emb = self.token_embedding_table(idx)
        # # [B,T,n_embd]
        #
        # positions = torch.arange(
        #     T,
        #     device=idx.device,
        # )
        #
        # pos_emb = self.position_embedding_table(
        #     positions
        # )
        # # [T,n_embd]
        #
        # x = tok_emb + pos_emb
        # # [B,T,n_embd]

        x = self.token_embedding_table(idx)

        # x = self.block(x)
        # [B,T,n_embd]

        for block in self.blocks:
            x = block(x)
            # [B, T, n_embd]

        x = self.rms_f(x)
        # [B,T,n_embd]

        logits = self.lm_head(x)
        # [B,T,vocab_size]

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape

            logits_for_loss = logits.reshape(
                B * T,
                C,
            )

            targets_for_loss = targets.reshape(
                B * T,
            )

            loss = F.cross_entropy(
                logits_for_loss,
                targets_for_loss,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
    ):
        for _ in range(max_new_tokens):

            idx_cond = idx[
                :,
                -self.block_size:
            ]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]
            # [B,vocab_size]

            probs = F.softmax(
                logits,
                dim=-1,
            )

            idx_next = torch.multinomial(
                probs,
                num_samples=1,
            )
            # [B,1]

            idx = torch.cat(
                (idx, idx_next),
                dim=1,
            )

        return idx


# =========================
# 创建模型
# =========================

if num_kv_heads == num_heads:
    attention_type = "MHA"

elif num_kv_heads == 1:
    attention_type = "MQA"

else:
    attention_type = "GQA"

print(
    f"Attention type: "
    f"{attention_type}"
)

print(
    f"Query heads: "
    f"{num_heads}"
)

print(
    f"KV heads: "
    f"{num_kv_heads}"
)

model = MiniGPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout,
).to(device)

print(
    f"Embedding table is shared with lm_head: " +
    str(model.lm_head.weight
    is model.token_embedding_table.weight)
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
)


# =========================
# 训练
# =========================

model.train()

for step in range(max_iters):
    xb, yb = get_batch("train")

    optimizer.zero_grad()

    logits, loss = model(
        xb,
        yb,
    )

    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(
            f"step {step}, "
            f"loss: {loss.item():.4f}"
        )


# =========================
# 生成
# =========================

model.eval()

context = torch.zeros(
    (1, 1),
    dtype=torch.long,
    device=device,
)

generated = model.generate(
    context,
    max_new_tokens=200,
)

print(
    decode(
        generated[0].tolist()
    )
)