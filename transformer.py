import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 超参数
# =========================

vocab_size = 32

# 0 专门作为 Decoder 的起始标记
bos_id = 0

seq_len = 6
max_seq_len = 16

d_model = 64
num_heads = 4
num_layers = 2
d_ff = 4 * d_model
dropout = 0

batch_size = 64
learning_rate = 1e-3
max_iters = 3000

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

torch.manual_seed(1337)


# =========================
# 固定长度反转任务
# =========================

def get_batch():
    # 0 留给 BOS。
    # 普通 token 从 1 到 vocab_size - 1。
    src = torch.randint(
        low=1,
        high=vocab_size,
        size=(batch_size, seq_len),
    )
    # [B,S]

    # 目标序列是源序列的反转
    tgt_out = torch.flip(
        src,
        dims=[1],
    )
    # [B,T]

    bos = torch.full(
        size=(batch_size, 1),
        fill_value=bos_id,
        dtype=torch.long,
    )
    # [B,1]

    # Teacher Forcing：
    #
    # 正确答案：
    # [4,5,9,2,7,3]
    #
    # Decoder 输入：
    # [BOS,4,5,9,2,7]
    #
    # 训练目标：
    # [4,5,9,2,7,3]
    tgt_in = torch.cat(
        [
            bos,
            tgt_out[:, :-1],
        ],
        dim=1,
    )
    # [B,T]

    return (
        src.to(device),
        tgt_in.to(device),
        tgt_out.to(device),
    )


# =========================
# 正弦位置编码
# =========================

class SinusoidalPositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_seq_len,
        dropout,
    ):
        super().__init__()

        if d_model % 2 != 0:
            raise ValueError(
                "d_model 必须为偶数"
            )

        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
        ).unsqueeze(1)
        # [max_seq_len,1]

        dimension_indices = torch.arange(
            0,
            d_model,
            2,
            dtype=torch.float32,
        )
        # [d_model/2]
        #
        # 例如 d_model=8：
        # [0,2,4,6]

        inv_freq = torch.exp(
            -math.log(10000.0)
            * dimension_indices
            / d_model
        )
        # [d_model/2]
        #
        # 等价于：
        # 1 / 10000^(2i/d_model)

        angles = positions * inv_freq
        # [max_seq_len,d_model/2]

        pe = torch.zeros(
            max_seq_len,
            d_model,
        )
        # [max_seq_len,d_model]

        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        self.register_buffer(
            "pe",
            pe,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):
        # x: [B,T,d_model]

        T = x.shape[1]

        if T > self.pe.shape[0]:
            raise ValueError(
                f"序列长度 T={T} 超过 "
                f"max_seq_len={self.pe.shape[0]}"
            )

        x = x + self.pe[:T].unsqueeze(0)
        # [B,T,C] + [1,T,C]
        # → [B,T,C]

        return self.dropout(x)


# =========================
# 通用多头注意力
# =========================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        dropout,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model 必须能被 num_heads 整除"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = (
            d_model // num_heads
        )

        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.k_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.v_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.attention_dropout = nn.Dropout(
            dropout
        )

        self.output_dropout = nn.Dropout(
            dropout
        )

    def split_heads(self, x):
        # x: [B,T,d_model]

        B, T, C = x.shape

        x = x.reshape(
            B,
            T,
            self.num_heads,
            self.head_dim,
        )
        # [B,T,H,D]

        x = x.transpose(1, 2)
        # [B,H,T,D]

        return x

    def merge_heads(self, x):
        # x: [B,H,T,D]

        B, H, T, D = x.shape

        x = x.transpose(1, 2)
        # [B,T,H,D]

        x = x.contiguous().view(
            B,
            T,
            H * D,
        )
        # [B,T,d_model]

        return x

    def forward(
        self,
        query_input,
        key_value_input,
        causal=False,
    ):
        # query_input:
        # [B,T_q,d_model]
        #
        # key_value_input:
        # [B,T_kv,d_model]

        q = self.q_proj(query_input)
        k = self.k_proj(key_value_input)
        v = self.v_proj(key_value_input)

        # q: [B,T_q,d_model]
        # k: [B,T_kv,d_model]
        # v: [B,T_kv,d_model]

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # q: [B,H,T_q,D]
        # k: [B,H,T_kv,D]
        # v: [B,H,T_kv,D]

        scores = (
            q @ k.transpose(-2, -1)
        )
        # [B,H,T_q,D]
        # @
        # [B,H,D,T_kv]
        #
        # →
        # [B,H,T_q,T_kv]

        scores = scores * (
            self.head_dim ** -0.5
        )

        if causal:
            T_q = q.shape[-2]
            T_kv = k.shape[-2]

            if T_q != T_kv:
                raise ValueError(
                    "当前 causal self-attention "
                    "要求 T_q == T_kv"
                )

            causal_mask = torch.tril(
                torch.ones(
                    T_q,
                    T_kv,
                    dtype=torch.bool,
                    device=scores.device,
                )
            )
            # [T_q,T_kv]

            causal_mask = causal_mask[
                None,
                None,
                :,
                :,
            ]
            # [1,1,T_q,T_kv]

            scores = scores.masked_fill(
                ~causal_mask,
                float("-inf"),
            )

        weights = F.softmax(
            scores,
            dim=-1,
        )
        # [B,H,T_q,T_kv]

        weights = self.attention_dropout(
            weights
        )

        out = weights @ v
        # [B,H,T_q,T_kv]
        # @
        # [B,H,T_kv,D]
        #
        # →
        # [B,H,T_q,D]

        out = self.merge_heads(out)
        # [B,T_q,d_model]

        out = self.out_proj(out)
        # [B,T_q,d_model]

        out = self.output_dropout(out)

        return out


# =========================
# 原始 Transformer FFN
# =========================

class FeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff,
        dropout,
    ):
        super().__init__()

        self.linear1 = nn.Linear(
            d_model,
            d_ff,
        )

        self.linear2 = nn.Linear(
            d_ff,
            d_model,
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):
        # [B,T,d_model]

        x = self.linear1(x)
        # [B,T,d_ff]

        x = F.relu(x)
        # [B,T,d_ff]

        x = self.linear2(x)
        # [B,T,d_model]

        x = self.dropout(x)

        return x


# =========================
# Encoder Layer
# =========================

class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout,
    ):
        super().__init__()

        self.self_attention = (
            MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
        )

        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        # 原始 Transformer 使用 Post-Norm
        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

    def forward(self, x):
        # x: [B,S,d_model]

        attention_out = (
            self.self_attention(
                query_input=x,
                key_value_input=x,
                causal=False,
            )
        )
        # [B,S,d_model]

        # Post-Norm：
        # 先残差相加，再 LayerNorm
        x = self.norm1(
            x + attention_out
        )

        feed_forward_out = (
            self.feed_forward(x)
        )
        # [B,S,d_model]

        x = self.norm2(
            x + feed_forward_out
        )

        return x


# =========================
# Decoder Layer
# =========================

class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout,
    ):
        super().__init__()

        self.self_attention = (
            MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
        )

        self.cross_attention = (
            MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
        )

        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(
            d_model
        )

        self.norm2 = nn.LayerNorm(
            d_model
        )

        self.norm3 = nn.LayerNorm(
            d_model
        )

    def forward(
        self,
        x,
        encoder_memory,
    ):
        # x:
        # [B,T,d_model]
        #
        # encoder_memory:
        # [B,S,d_model]

        # -------------------------
        # 1. Causal Self-Attention
        # -------------------------

        self_attention_out = (
            self.self_attention(
                query_input=x,
                key_value_input=x,
                causal=True,
            )
        )
        # [B,T,d_model]

        x = self.norm1(
            x + self_attention_out
        )

        # -------------------------
        # 2. Cross-Attention
        # -------------------------

        cross_attention_out = (
            self.cross_attention(
                # Q 来自 Decoder
                query_input=x,

                # K、V 来自 Encoder
                key_value_input=encoder_memory,

                causal=False,
            )
        )
        # [B,T,d_model]

        x = self.norm2(
            x + cross_attention_out
        )

        # -------------------------
        # 3. Feed Forward
        # -------------------------

        feed_forward_out = (
            self.feed_forward(x)
        )

        x = self.norm3(
            x + feed_forward_out
        )

        return x


# =========================
# 完整 Encoder–Decoder
# =========================

class OriginalTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_len,
        dropout,
    ):
        super().__init__()

        self.d_model = d_model

        self.src_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.tgt_embedding = nn.Embedding(
            vocab_size,
            d_model,
        )

        self.src_position = (
            SinusoidalPositionalEncoding(
                d_model=d_model,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
        )

        self.tgt_position = (
            SinusoidalPositionalEncoding(
                d_model=d_model,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
        )

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.output_projection = nn.Linear(
            d_model,
            vocab_size,
        )

    def encode(self, src):
        # src: [B,S]

        x = self.src_embedding(src)
        # [B,S,d_model]

        # 原始 Transformer 会把 embedding
        # 乘以 sqrt(d_model)
        x = x * math.sqrt(
            self.d_model
        )

        x = self.src_position(x)
        # [B,S,d_model]

        for layer in self.encoder_layers:
            x = layer(x)

        return x
        # Encoder memory:
        # [B,S,d_model]

    def decode(
        self,
        tgt_in,
        encoder_memory,
    ):
        # tgt_in: [B,T]

        x = self.tgt_embedding(tgt_in)
        # [B,T,d_model]

        x = x * math.sqrt(
            self.d_model
        )

        x = self.tgt_position(x)
        # [B,T,d_model]

        for layer in self.decoder_layers:
            x = layer(
                x=x,
                encoder_memory=encoder_memory,
            )

        return x
        # [B,T,d_model]

    def forward(
        self,
        src,
        tgt_in,
        targets=None,
    ):
        encoder_memory = self.encode(src)
        # [B,S,d_model]

        decoder_hidden = self.decode(
            tgt_in=tgt_in,
            encoder_memory=encoder_memory,
        )
        # [B,T,d_model]

        logits = self.output_projection(
            decoder_hidden
        )
        # [B,T,vocab_size]

        if targets is None:
            loss = None

        else:
            B, T, V = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, src):
        # src: [B,S]

        encoder_memory = self.encode(src)
        # Encoder 只需要计算一次

        B = src.shape[0]

        generated = torch.full(
            size=(B, 1),
            fill_value=bos_id,
            dtype=torch.long,
            device=src.device,
        )
        # 开始时只有：
        # [BOS]

        # 固定长度任务：
        # 输入有多少个 token，就生成多少个 token。
        for _ in range(src.shape[1]):
            decoder_hidden = self.decode(
                tgt_in=generated,
                encoder_memory=encoder_memory,
            )

            last_hidden = (
                decoder_hidden[:, -1, :]
            )
            # [B,d_model]

            logits = self.output_projection(
                last_hidden
            )
            # [B,vocab_size]

            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True,
            )
            # [B,1]

            generated = torch.cat(
                [
                    generated,
                    next_token,
                ],
                dim=1,
            )

        # 去掉开头的 BOS
        return generated[:, 1:]


# =========================
# 创建模型
# =========================

model = OriginalTransformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    max_seq_len=max_seq_len,
    dropout=dropout,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"参数量：{num_parameters:,}"
)


# =========================
# 训练
# =========================

model.train()

for step in range(max_iters):
    src, tgt_in, tgt_out = get_batch()

    optimizer.zero_grad()

    logits, loss = model(
        src=src,
        tgt_in=tgt_in,
        targets=tgt_out,
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    if step % 100 == 0:
        print(
            f"step {step}, "
            f"loss: {loss.item():.4f}"
        )


# =========================
# 自回归生成
# =========================

model.eval()

test_src = torch.tensor(
    [
        [3, 7, 2, 9, 5, 4],
        [8, 1, 6, 6, 2, 10],
    ],
    dtype=torch.long,
    device=device,
)

predicted = model.generate(
    test_src
)

print("source:")
print(test_src.cpu())

print("expected:")
print(
    torch.flip(
        test_src,
        dims=[1],
    ).cpu()
)

print("predicted:")
print(predicted.cpu())