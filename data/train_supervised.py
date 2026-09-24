# data/train_supervised.py
"""
Train the neural network on Stockfish-generated data.
Uses curriculum learning — train on easier games first,
then harder games, so the network builds up knowledge
progressively.
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.net import ChessNet, load_model
from tqdm import tqdm


def load_dataset(path):
    """Load a saved Stockfish dataset."""
    if not os.path.exists(path):
        print(f"⚠ Dataset not found: {path}")
        return None, None
    data = torch.load(path)
    print(f"✓ Loaded {len(data['tensors'])} positions from {path}")
    print(f"  ELO: {data.get('stockfish_elo', 'unknown')} | "
          f"Games: {data.get('n_games', 'unknown')} | "
          f"Results: {data.get('results', {})}")
    return data['tensors'], data['labels']


def train_on_dataset(model, tensors, labels, epochs=10,
                     batch_size=512, lr=1e-3, device='cpu',
                     dataset_name=''):
    """Train model on a single dataset."""

    print(f"\n{'='*50}")
    print(f"Training on: {dataset_name}")
    print(f"  Positions: {len(tensors)}")
    print(f"  Epochs:    {epochs}")
    print(f"  LR:        {lr}")
    print(f"  Device:    {device}")
    print(f"{'='*50}")

    tensors = tensors.to(device)
    labels  = labels.to(device)

    dataset = TensorDataset(tensors, labels)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs)
    loss_fn   = nn.MSELoss()

    model.train()
    model.to(device)

    best_loss  = float('inf')
    best_state = None

    for epoch in tqdm(range(epochs), desc=f"🧠 {dataset_name}"):
        total_loss = 0

        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            output = model(X)
            # Handle both single-head and dual-head
            if isinstance(output, tuple):
                value = output[0]
            else:
                value = output

            loss = loss_fn(value, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        tqdm.write(f"  Epoch {epoch+1}/{epochs}  loss: {avg_loss:.4f}")

        # Save best model state
        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}

        # Early stop if loss is excellent
        if avg_loss < 0.01:
            tqdm.write("  ⚡ Early stop — loss excellent")
            break

    # Restore best state
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  ✓ Restored best state — loss: {best_loss:.4f}")

    return model


def train_curriculum(device='cpu'):
    """
    Curriculum learning:
    1. Load existing checkpoint if available
    2. Train on ELO 1350 (entry level Stockfish)
    3. Train on ELO 1500 (intermediate)
    4. Train on ELO 1800 (stronger)
    5. Save final model
    """

    # ── Load or create model ──
    checkpoint = 'checkpoints/chess_net_latest.pth'
    if os.path.exists(checkpoint):
        model = load_model(checkpoint, device=device)
        print(f"✓ Loaded existing checkpoint: {checkpoint}")
    else:
        model = ChessNet().to(device)
        print("Starting fresh model")

    os.makedirs('../checkpoints', exist_ok=True)

    # ── Stage 1: ELO 1350 — entry level ──
    X1, y1 = load_dataset('data/stockfish_1350.pt')
    if X1 is not None:
        model = train_on_dataset(
            model, X1, y1,
            epochs=15,
            batch_size=512,
            lr=1e-3,
            device=device,
            dataset_name='ELO 1350 (entry level)'
        )
        torch.save(model.state_dict(),
                   'checkpoints/supervised_stage1.pth')
        print("✓ Stage 1 checkpoint saved")
    else:
        print("⚠ Skipping Stage 1 — dataset not found")

    # ── Stage 2: ELO 1500 — intermediate ──
    X2, y2 = load_dataset('data/stockfish_1500.pt')
    if X2 is not None:
        model = train_on_dataset(
            model, X2, y2,
            epochs=15,
            batch_size=512,
            lr=5e-4,   # lower LR for refinement
            device=device,
            dataset_name='ELO 1500 (intermediate)'
        )
        torch.save(model.state_dict(),
                   'checkpoints/supervised_stage2.pth')
        print("✓ Stage 2 checkpoint saved")
    else:
        print("⚠ Skipping Stage 2 — dataset not found")

    # ── Stage 3: ELO 1800 — stronger ──
    X3, y3 = load_dataset('data/stockfish_1800.pt')
    if X3 is not None:
        model = train_on_dataset(
            model, X3, y3,
            epochs=10,
            batch_size=512,
            lr=1e-4,   # fine-tuning LR
            device=device,
            dataset_name='ELO 1800 (stronger)'
        )
        torch.save(model.state_dict(),
                   'checkpoints/supervised_stage3.pth')
        print("✓ Stage 3 checkpoint saved")
    else:
        print("⚠ Skipping Stage 3 — dataset not found")

    # ── Save final model ──
    torch.save(model.state_dict(),
               'checkpoints/chess_net_latest.pth')
    torch.save(model.state_dict(),
               'checkpoints/chess_net_supervised.pth')

    print("\n" + "="*50)
    print("✓ Supervised training complete!")
    print("  Checkpoints saved:")
    print("  → checkpoints/supervised_stage1.pth")
    print("  → checkpoints/supervised_stage2.pth")
    print("  → checkpoints/supervised_stage3.pth")
    print("  → checkpoints/chess_net_supervised.pth")
    print("  → checkpoints/chess_net_latest.pth")
    print("="*50)
    print("\nNext steps:")
    print("  1. Test: python3 webapp/app.py")
    print("  2. Upload to Kaggle for more self-play training")

    return model


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    print(f"✓ Running from: {os.getcwd()}\n")
    train_curriculum(device=device)