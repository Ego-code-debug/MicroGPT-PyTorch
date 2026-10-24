import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class MicroGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, block_size):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), 
            nn.ReLU(), 
            nn.Linear(4 * n_embd, n_embd)
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
        
        # 【核心修复：GPU 张量向量化重复惩罚】
        # 1. 找出当前已经生成的所有 token 中，不重复的索引
            unique_tokens = torch.unique(idx[0])
        
        # 2. 提取这些重复 token 对应的 logits
            token_logits = logits[0, unique_tokens]
        
        # 3. 利用 torch.where 在 GPU 上并行惩罚（大于0除以2.0，小于0乘以2.0）
        # 这里把惩罚系数从 1.5 提高到了 2.0，加大打击力度！
            penalized_logits = torch.where(token_logits > 0, token_logits / 2.0, token_logits * 1.2)
        
            logits[0, unique_tokens] = penalized_logits
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx