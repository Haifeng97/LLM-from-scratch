import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 超参数
# =========================

n_embd = 32
num_heads = 4
n_layer = 4
block_size = 16
dropout = 0.1

learning_rate = 3e-3
max_iters = 3000
batch_size = 16

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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

        # x: [B, T, head_size]

        B, T, D = x.shape

        # even / odd here references to the e/o of indexes in python, which starts from 0 (0 is even)
        x_even = x[:, :, 0::2]
        x_odd = x[..., 1::2]

        # cos, sin: [block_size, head_size / 2] ----[:T].unsqueeze(0)----> [1, T, head_size / 2]
        cos = self.cos[:T].unsqueeze(0)
        sin = self.sin[:T].unsqueeze(0)

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        out = torch.stack(
            [
                rotated_even,
                rotated_odd
            ],
            dim = -1
        )
        # After stack: [B, T, head_size / 2, 2]

        out = out.flatten(-2)
        # After flattening: [B, T, head_size]

        return out








# =========================
# 单个注意力头
# =========================

class Head(nn.Module):

    def __init__(
        self,
        n_embd,
        head_size,
        block_size,
        dropout,
    ):
        super().__init__()

        self.query = nn.Linear(
            n_embd,
            head_size,
            bias=False,
        )

        self.key = nn.Linear(
            n_embd,
            head_size,
            bias=False,
        )

        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False,
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size,
                )
            ),
        )

        self.rope = RotaryEmbedding(head_size = head_size, block_size = block_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B,T,n_embd]

        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # [B,T,head_size]

        q = self.rope(q)
        k = self.rope(k)

        scores = q @ k.transpose(-2, -1)
        # [B,T,T]

        scores = scores * (
            k.shape[-1] ** -0.5
        )

        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf"),
        )

        weights = F.softmax(
            scores,
            dim=-1,
        )
        # [B,T,T]

        weights = self.dropout(weights)

        out = weights @ v
        # [B,T,head_size]

        return out


# =========================
# 多头注意力
# =========================

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        n_embd,
        num_heads,
        block_size,
        dropout,
    ):
        super().__init__()

        if n_embd % num_heads != 0:
            raise ValueError(
                "n_embd 必须能被 num_heads 整除"
            )

        head_size = n_embd // num_heads

        self.heads = nn.ModuleList([
            Head(
                n_embd=n_embd,
                head_size=head_size,
                block_size=block_size,
                dropout=dropout,
            )
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(
            n_embd,
            n_embd,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 每个头：
        # [B,T,n_embd] -> [B,T,head_size]

        out = torch.cat(
            [
                head(x)
                for head in self.heads
            ],
            dim=-1,
        )
        # [B,T,n_embd]

        out = self.proj(out)
        # [B,T,n_embd]

        out = self.dropout(out)

        return out


# =========================
# FeedForward / MLP
# =========================

class FeedForward(nn.Module):

    def __init__(
        self,
        n_embd,
        dropout,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                n_embd,
                4 * n_embd,
            ),

            nn.GELU(),

            nn.Linear(
                4 * n_embd,
                n_embd,
            ),

            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B,T,n_embd]
        return self.net(x)
        # 输出仍然是 [B,T,n_embd]


# =========================
# Transformer Block
# =========================

class Block(nn.Module):

    def __init__(
        self,
        n_embd,
        num_heads,
        block_size,
        dropout,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)

        self.self_attention = MultiHeadAttention(
            n_embd=n_embd,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(n_embd)

        self.feed_forward = FeedForward(
            n_embd=n_embd,
            dropout=dropout,
        )

    def forward(self, x):
        # 第一条残差分支：注意力
        x = x + self.self_attention(
            self.ln1(x)
        )

        # 第二条残差分支：FFN
        x = x + self.feed_forward(
            self.ln2(x)
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
            Block(n_embd = n_embd, num_heads = num_heads, block_size = block_size, dropout = dropout)
            for _ in range(n_layer)
        ])

        # 所有 Block 后面的最终 LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)

        # hidden state -> 词表 logits
        self.lm_head = nn.Linear(
            n_embd,
            vocab_size,
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

        x = self.ln_f(x)
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

model = MiniGPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    num_heads=num_heads,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout,
).to(device)

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