# model/train.py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model.net import ChessNet
from tqdm import tqdm
import os

def train(epochs=20, batch_size=256, lr=1e-3, data_path='data/training_data.pt'):
    data = torch.load(data_path)
    X, y = data['tensors'], data['labels']

    print(f"📦 Loaded {len(X)} positions for training")

    dataset = TensorDataset(X, y)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ChessNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in tqdm(range(epochs), desc="🧠 Training", unit="epoch"):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        tqdm.write(f"  Epoch {epoch+1}/{epochs}  loss: {avg_loss:.4f}")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/chess_net_v1.pth')
    print("\n✓ Model saved to checkpoints/chess_net_v1.pth")
    return model