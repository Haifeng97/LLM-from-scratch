import torch
import torch.nn as nn
import torch.nn.functional as F


n_embd = 32
vocab_size = 512
block_size = 16
# learning_rate = 1e-2
learning_rate = 3e-3
max_iters = 1000
torch.manual_seed(1337)
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text = """
今天天气很好，我想出去走走。
我喜欢学习人工智能，也喜欢研究大模型。
大模型可以根据前面的文字预测后面的文字。
我们现在正在从零开始写一个小语言模型。
"""

# 建立字符级词表
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 字符 <-> id
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join([itos[i] for i in ids])

# 把整段文本变成 token id
data = torch.tensor(encode(text), dtype=torch.long)

# 切分训练集和验证集
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    source = train_data if split == "train" else val_data

    # 随机取 batch_size 个起点
    ix = torch.randint(len(source) - block_size, (batch_size,))

    # x 是输入，y 是目标
    # y 比 x 向后错一位
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])

    x = x.to(device)
    y = y.to(device)

    return x, y

class Head(nn.Module):

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()

        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):

        B, T, D = x.shape

        q = self.query(x) # [B, T, head_size]
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.mT # [B, T, head_size] @ [B, head_size, T] -> [B, T, T]
        scores = scores * k.shape[-1] ** -0.5

        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        weights = F.softmax(scores, dim = -1) # [B, T, T]

        out = weights @ v # [B, T, T] @ [B, T, head_size] -> [B, T, head_size]

        return out


class AttentionLM(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()

        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.sa_head = Head(n_embd = n_embd, head_size = n_embd, block_size = block_size)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # [B, T, n_embd]

        positions = torch.arange(T, device = idx.device) # [T]

        pe = self.position_embedding_table(positions)

        x = tok_emb + pe # [B, T, n_embd]
        x = self.sa_head(x)

        logits = self.lm_head(x) # [B, T, vocab_size]

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits_for_loss = logits.reshape(B *T, C)
            targets_for_loss = targets.reshape(B * T)

            loss = F.cross_entropy(logits_for_loss, targets_for_loss)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] # [B, vocab_size]
            probs = F.softmax(logits, dim = -1)

            idx_next = torch.multinomial(probs, num_samples = 1) # [B, 1]
            # idx_next = torch.argmax(probs, dim = 1, keepdim=True) # [B, 1]

            idx = torch.cat((idx, idx_next), dim = 1) # [B, <original_length> + 1]

        return idx



model = AttentionLM(vocab_size = vocab_size, n_embd = n_embd, block_size = block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

# training loops
for step in range(max_iters):
    xb, yb = get_batch("train")
    optimizer.zero_grad()
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(
            f'step {step}, '
            f'loss: {loss.item(): .4f}'
        )

context = torch.zeros((1, 1), dtype = torch.long, device = device)
generated = model.generate(context, max_new_tokens = 1000)
print(decode(generated[0].tolist()))