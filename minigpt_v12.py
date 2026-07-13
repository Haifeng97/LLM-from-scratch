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

    def forward(self, x, position_offset = 0):

        # x: [B, H, T, head_size]

        B, H, T, D = x.shape

        position_end = position_offset + T
        if position_end > self.cos.shape[0]:
            raise ValueError(
                f'RoPE position ends = {position_end} exceeds block_size = {self.cos.shape[0]}'
            )

        # even / odd here references to the e/o of indexes in python, which starts from 0 (0 is even)
        # [B, H, T, head_size / 2]
        x_even = x[:, :, :, 0::2]
        x_odd = x[..., 1::2]

        # Prefill:
        # position_offset = 0
        # T = prompt length
        # Decode:
        # position_offset = past_length
        # T = 1

        # cos, sin: [block_size, head_size / 2] ----[:T][None, None, :, :]----> [1, 1, T, head_size / 2]
        cos = self.cos[position_offset: position_end][None, None, :, :]
        sin = self.sin[position_offset: position_end][None, None, :, :]

        cos = cos.to(device = x.device, dtype=x.dtype)
        sin = sin.to(device = x.device, dtype=x.dtype)

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

    def forward(
            self, x, past_key_value = None, use_cache = False
    ):
        # x: [B,T,n_embd]
        #
        # past_key_value: (past_key, past_value)
        #
        # past_key:
        # [B, num_kv_heads, past_len, head_size]
        #
        # past_value:
        # [B, num_kv_heads, past_len, head_size]

        B, T, C = x.shape

        if past_key_value is None:
            past_length = 0

            past_key = None
            past_value = None

        else:
            past_key, past_value = past_key_value
            past_length = past_key.shape[2]

            if T != 1:
                raise ValueError(
                    "when using KV Cache, decode currently requires T == 1"
                )

        total_length = past_length + T
        if total_length > self.block_size:
            raise ValueError(
                f'total sequence length {total_length} exceeds block_size = {self.block_size}'
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


        q = self.rope(q, position_offset = past_length)
        k = self.rope(k, position_offset = past_length)

        if past_key is not None:
            k = torch.cat(
                [past_key, k],
                dim = 2
            )

            v = torch.cat(
                [past_value, v],
                dim = 2
            )
            # k / v:
            # [B, H, total_length, D]

        if use_cache:
            present_key_value = (
                k, v
            )

        else:
            present_key_value = None

        k_for_attention = (
            k.repeat_interleave(self.num_kv_groups, dim = 1)
        )

        v_for_attention = (
            v.repeat_interleave(self.num_kv_groups, dim = 1)
        )
        # qkv: [B, H, T, D]

        attention_dropout_p = self.dropout_p if self.training else 0.0

        is_causal = past_key is None

        out = F.scaled_dot_product_attention(
            query=q,
            key=k_for_attention,
            value=v_for_attention,
            attn_mask=None,
            dropout_p=attention_dropout_p,
            is_causal=is_causal
        )
        # [B, H, T, D]

        out = out.transpose(1, 2)
        # [B, T, H, D]

        out = out.contiguous().view(B, T, C)

        out = self.out_proj(out)
        # [B, T, n_embd]

        out = self.out_dropout(out)

        return (
            out, present_key_value
        )


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

    def forward(self, x, past_key_value = None, use_cache = False):

        attention_out, present_key_value = (
            self.self_attention(
                self.rms1(x),
                past_key_value = past_key_value,
                use_cache = use_cache
            )
        )

        x = x + attention_out
        x = x + self.feed_forward(
            self.rms2(x)
        )

        return (
            x, present_key_value
        )


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
        past_key_values = None,
        use_cache = False,
    ):

        # idx: [B, T] when training or prefilling
        #      [B, 1] when decoding

        B, T = idx.shape

        if past_key_values is None:
            past_key_values = [
                None
                for _ in self.blocks
            ]
        else:
            if len(past_key_values) != len(self.blocks):
                raise ValueError(
                    'past_key_values length must equal n_layer'
                )

        if targets is not None and any(cache is not None for cache in past_key_values):
            raise ValueError(
                'Training with past cache is not supported'
            )


        x = self.token_embedding_table(idx)

        # x = self.block(x)
        # [B,T,n_embd]

        new_past_key_values = []

        for block, layer_past in zip(self.blocks, past_key_values):
            x, layer_present = block(
                x, past_key_value = layer_past, use_cache = use_cache
            )

            if use_cache:
                new_past_key_values.append(layer_present)

        x = self.rms_f(x)
        # [B,T,n_embd]

        logits = self.lm_head(x)
        # [B,T,vocab_size]

        if targets is None:
            loss = None

        else:
            B, T, V = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T)
            )

        if use_cache:
            cache_output = new_past_key_values
        else:
            cache_output = None

        return logits, loss, cache_output

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
    ):
        total_length = idx.shape[1] + max_new_tokens

        if total_length > self.block_size:
            raise ValueError(
                f'prompt length {idx.shape[1]} + max_new_tokens {max_new_tokens} > block size {self.block_size}'
            )

        # Prefill
        logits, _, past_key_values = self(idx, use_cache = True)
        # Cache is containing K/V of all tokens

        for layer_index, (
                key_cache,
                value_cache,
        ) in enumerate(
            past_key_values
        ):
            print(
                f"Layer {layer_index}:"
            )

            print(
                "K cache:",
                key_cache.shape,
            )

            print(
                "V cache:",
                value_cache.shape,
            )



        # Decode
        for step in range(max_new_tokens):

            last_logits = logits[:, -1, :]

            probs = F.softmax(
                last_logits, dim = -1
            )

            idx_next = torch.multinomial(
                probs, num_samples=1,
            )

            # [B, 1]

            idx = torch.cat(
                [idx, idx_next],
                dim = 1
            )

            if step == max_new_tokens - 1:
                break

            logits, _, past_key_values = self(
                idx = idx_next,
                past_key_values = past_key_values,
                use_cache = True
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

    logits, loss, _ = model(
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
    max_new_tokens=15,
)

print(
    decode(
        generated[0].tolist()
    )
)