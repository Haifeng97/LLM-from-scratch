import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 超参数
# =========================

n_embd = 32
num_heads = 4
block_size = 16

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


# =========================
# 单个注意力头
# =========================

class Head(nn.Module):

    def __init__(
        self,
        n_embd,
        head_size,
        block_size,
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

    def forward(self, x):
        # x: [B,T,n_embd]

        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # q、k、v:
        # [B,T,head_size]

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
            )
            for _ in range(num_heads)
        ])

        # 把多个头的信息进一步混合
        self.proj = nn.Linear(
            n_embd,
            n_embd,
        )

    def forward(self, x):
        # 每个 head 都接收完整的 x
        # 每个 head 输出 [B,T,head_size]

        head_outputs = [
            head(x)
            for head in self.heads
        ]

        # 沿最后一个维度拼接
        out = torch.cat(
            head_outputs,
            dim=-1,
        )
        # [B,T,num_heads*head_size]
        # 也就是 [B,T,n_embd]

        out = self.proj(out)
        # [B,T,n_embd]

        return out



# MQA
# class MultiQueryAttention(nn.Module):
#     def __init__(self, n_embd, head_size, block_size):
#         super().__init__()
#
#         self.head_num = n_embd / head_size
# todo: MQA / GQA




# =========================
# 多头注意力语言模型
# =========================

class MultiHeadAttentionLM(nn.Module):

    def __init__(
        self,
        vocab_size,
        n_embd,
        num_heads,
        block_size,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd,
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd,
        )

        self.self_attention = MultiHeadAttention(
            n_embd=n_embd,
            num_heads=num_heads,
            block_size=block_size,
        )

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

        tok_emb = self.token_embedding_table(idx)
        # [B,T,n_embd]

        positions = torch.arange(
            T,
            device=idx.device,
        )

        pos_emb = self.position_embedding_table(
            positions
        )
        # [T,n_embd]

        x = tok_emb + pos_emb
        # [B,T,n_embd]

        x = self.self_attention(x)
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
                B * T
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
            # idx_next = torch.argmax(probs, dim = -1, keepdim = True)
            # [B,1]

            idx = torch.cat(
                (idx, idx_next),
                dim=1,
            )

        return idx


# =========================
# 创建模型
# =========================

model = MultiHeadAttentionLM(
    vocab_size=vocab_size,
    n_embd=n_embd,
    num_heads=num_heads,
    block_size=block_size,
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