import torch
import time
import math
from data import get_batch, vocab_size, itos, block_size, device
from model import MicroGPT

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = []
        for _ in range(50):
            x, y = get_batch(split)
            if x is not None:
                _, loss = model(x, y)
                losses.append(loss.item())
        out[split] = sum(losses) / len(losses) if losses else 0.0
    model.train()
    return out

if __name__ == "__main__":
    n_embd = 64
    n_head = 4
    
    model = MicroGPT(vocab_size, n_embd, n_head, block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print(f"Using device: {device}")

    start_time = time.time()
    max_steps = 10000
    
    for step in range(max_steps):
        x, y = get_batch('train')
        if x is None: continue
            
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 1000 == 0 and step > 0:
            losses = estimate_loss(model)
            print(f"step {step} | train_loss {losses['train']:.4f} | val_loss {losses['val']:.4f}")

    end_time = time.time()
    
    print("\n" + "="*30)
    print("[Part 2] 验收指标：")
    
    losses = estimate_loss(model)
    val_loss = losses['val']
    perplexity = math.exp(val_loss) if val_loss < 20 else float('inf')
    print(f"最终验证集 Loss: {val_loss:.4f}")
    print(f"验证集困惑度 (Perplexity): {perplexity:.2f}")

    model.eval()
    start_tokens = torch.randint(0, vocab_size, (1, 1), dtype=torch.long, device=device)
    gen_start_time = time.time()
    generated_ids = model.generate(start_tokens, max_new_tokens=100)
    gen_end_time = time.time()

    gen_text = ''.join([itos[i] for i in generated_ids[0].tolist()])
    print(f"生成的示例名字:")
    for i in range(0, len(gen_text), 6):
        print(f"- {gen_text[i:i+6]}")

    total_time = gen_end_time - gen_start_time
    throughput = 100 / total_time
    print(f"\n推理生成 100 个 token 耗时: {total_time:.4f} 秒")
    print(f"推理吞吐量: {throughput:.2f} tokens/s")
    print("="*30)