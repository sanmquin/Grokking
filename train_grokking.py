import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class StandardTransformer(nn.Module):
    def __init__(self, p, d_model=128, num_heads=4, mlp_dim=512):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Vocab has p + 1 tokens (0 to p-1, and = token)
        self.tok_embed = nn.Embedding(p + 1, d_model)
        self.pos_embed = nn.Embedding(3, d_model)

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.mlp_in = nn.Linear(d_model, mlp_dim, bias=False)
        self.mlp_out = nn.Linear(mlp_dim, d_model, bias=False)

        # Unembedding layer maps to p possible outputs (0 to p-1)
        self.unembed = nn.Linear(d_model, p, bias=False)

    def forward(self, x):
        B, L = x.shape  # B, 3
        pos = torch.arange(L, device=x.device).unsqueeze(0)  # 1, 3
        h = self.tok_embed(x) + self.pos_embed(pos)  # B, 3, d_model

        # Attention
        Q = self.W_Q(h).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_K(h).view(B, L, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_V(h).view(B, L, self.num_heads, self.d_head).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_out = self.W_O((F.softmax(scores, dim=-1) @ V).transpose(1, 2).contiguous().view(B, L, self.d_model))

        h = h + attn_out

        # MLP
        h = h + self.mlp_out(F.relu(self.mlp_in(h)))

        # Unembed the final sequence position (index 2, corresponding to '=')
        return self.unembed(h[:, 2, :])

def make_dataset(p=113, frac_train=0.3, seed=42):
    set_seed(seed)
    all_pairs = [(a, b) for a in range(p) for b in range(p)]
    random.shuffle(all_pairs)

    n_train = int(len(all_pairs) * frac_train)

    # Input format: [a, b, p] where p represents the '=' token (index 113)
    train_x = torch.tensor([[a, b, p] for a, b in all_pairs[:n_train]], dtype=torch.long)
    train_y = torch.tensor([(a + b) % p for a, b in all_pairs[:n_train]], dtype=torch.long)

    test_x = torch.tensor([[a, b, p] for a, b in all_pairs[n_train:]], dtype=torch.long)
    test_y = torch.tensor([(a + b) % p for a, b in all_pairs[n_train:]], dtype=torch.long)

    return train_x, train_y, test_x, test_y

def train():
    p = 113
    frac_train = 0.3
    epochs = 5
    lr = 1e-3
    weight_decay = 1.0

    print(f"Generating dataset for modulo {p} addition with {frac_train*100}% train split...")
    train_x, train_y, test_x, test_y = make_dataset(p, frac_train, seed=42)
    print(f"Train size: {train_x.shape[0]}, Test size: {test_x.shape[0]}")

    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    model = StandardTransformer(p).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    for epoch in range(epochs + 1):
        model.train()
        logits = model(train_x)
        loss = criterion(logits, train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_acc = (logits.argmax(-1) == train_y).float().mean().item()

            test_logits = model(test_x)
            test_loss = criterion(test_logits, test_y).item()
            test_acc = (test_logits.argmax(-1) == test_y).float().mean().item()

            print(f"Epoch {epoch:5d} | Train Loss: {loss.item():.4e} | Test Loss: {test_loss:.4e} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    train()
