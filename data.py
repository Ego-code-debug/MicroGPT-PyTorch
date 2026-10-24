import torch
import random

block_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    with open('input.txt', 'r') as f:
        docs = [line.strip() for line in f if line.strip()]
    
    chars = sorted(list(set(''.join(docs))))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    random.shuffle(docs)
    n = int(0.8 * len(docs))
    train_docs, val_docs = docs[:n], docs[n:]
    
    return train_docs, val_docs, vocab_size, stoi, itos

train_docs, val_docs, vocab_size, stoi, itos = load_data()

def get_batch(split):
    data = train_docs if split == 'train' else val_docs
    if len(data) == 0:
        return None, None
    
    doc = random.choice(data)
    ids = [stoi[c] for c in doc if c in stoi]
    
    if len(ids) < block_size + 1:
        ids = ids * ((block_size + 1) // len(ids) + 1)
    ids = ids[:block_size + 1]
    
    x = torch.tensor(ids[:block_size], dtype=torch.long)
    y = torch.tensor(ids[1:block_size+1], dtype=torch.long)
    
    return x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)