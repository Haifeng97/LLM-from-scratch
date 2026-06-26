import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================
# 1. 一些超参数
# =====================

batch_size = 16       # 一次取多少段文本训练
block_size = 8        # 每段文本长度，也就是上下文长度
max_iters = 1000      # 训练多少步
eval_interval = 100
learning_rate = 1e-2

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)

# =====================
# 2. 准备一小段训练文本
# =====================

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

# =====================
# 3. 构造 batch
# =====================

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

# =====================
# 4. 定义 Bigram 语言模型
# =====================

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # 这个表的含义：
        # 每个 token id 直接查出“下一个 token 的 logits”
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx shape: [B, T]
        # B = batch size
        # T = sequence length

        logits = self.token_embedding_table(idx)
        # logits shape: [B, T, vocab_size]

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            # cross_entropy 要求输入是 [N, C]
            # target 是 [N]
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx shape: [B, T]
        for _ in range(max_new_tokens):
            logits, loss = self(idx)

            # 只取最后一个位置的 logits
            logits = logits[:, -1, :]
            # shape: [B, vocab_size]

            probs = F.softmax(logits, dim=-1)

            # 按概率采样下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)
            # idx_next = torch.argmax(probs, dim=1, keepdim=True)
            # shape: [B, 1]

            # 拼回原序列
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# =====================
# 5. 创建模型
# =====================

model = BigramLanguageModel(vocab_size)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# =====================
# 6. 训练
# =====================

for step in range(max_iters):
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        print(f"step {step}, loss {loss.item():.4f}")

# =====================
# 7. 生成文本
# =====================

# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# generated = model.generate(context, max_new_tokens=100)
#
# print("------ generated text ------")
# print(decode(generated[0].tolist()))

context = torch.tensor([encode(i) for i in ['我', '今', '天', '喜']], dtype=torch.long, device=device)


generated = model.generate(context, max_new_tokens=200)
# print([f'{decode(i.tolist())}\n' for i in generated])

print(
    *[f'{decode(i.tolist())}\n' for i in generated],
    sep="-------------------------\n"
)