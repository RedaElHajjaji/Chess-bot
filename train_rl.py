# train_rl.py
import torch
import os
import sys
import shutil
sys.path.insert(0, '/kaggle/working/Chess-bot')

from model.net import ChessNet, load_model
from self_play.loop import training_iteration
from engine.search import make_minimax_bot, make_neural_eval, tournament

# ── Auto-detect GPU ──
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✓ Running on: {device}")

# ── Load or create model ──
checkpoint = 'checkpoints/chess_net_latest.pth'
if os.path.exists(checkpoint):
    model = load_model(checkpoint, device=device)
    print("✓ Loaded existing checkpoint")
else:
    model = ChessNet().to(device)
    print("Starting fresh model")

os.makedirs('checkpoints', exist_ok=True)
os.makedirs('/kaggle/working/checkpoints', exist_ok=True)

# ── Training loop ──
N_ITERATIONS = 20

for i in range(1, N_ITERATIONS + 1):
    print(f"\n{'='*50}")
    print(f"ITERATION {i} / {N_ITERATIONS}")
    print(f"{'='*50}")

    model = training_iteration(
        model,
        iteration=i,
        games_per_iter=300,   # double for more diversity
        epochs=5,             # reduced from 10 to prevent memorization
        device=device,
        n_workers=4
    )

    # Save locally
    torch.save(model.state_dict(), f'checkpoints/chess_net_iter{i}.pth')
    torch.save(model.state_dict(), 'checkpoints/chess_net_latest.pth')

    # Save to Kaggle output folder so it persists after session ends
    shutil.copy('checkpoints/chess_net_latest.pth',
                f'/kaggle/working/chess_net_latest.pth')
    shutil.copy(f'checkpoints/chess_net_iter{i}.pth',
                f'/kaggle/working/chess_net_iter{i}.pth')
    print(f"✓ Saved to Kaggle output: iter{i}")

    # Benchmark every 5 iterations — neural vs its own past self
    if i % 5 == 0:
        neural_eval = make_neural_eval(model, device=device)
        neural_bot  = make_minimax_bot(depth=2, eval_fn=neural_eval)

        # Compare against iteration 1 as fixed baseline
        iter1_path = 'checkpoints/chess_net_iter1.pth'
        if os.path.exists(iter1_path):
            from model.net import load_model as lm
            early_model = lm(iter1_path, device='cpu')
            early_eval  = make_neural_eval(early_model, device='cpu')
            early_bot   = make_minimax_bot(depth=2, eval_fn=early_eval)
            print(f"\n--- Benchmark: iter{i} vs iter1 ---")
            tournament(neural_bot, early_bot, n_games=10)
        else:
            # Fallback to handcrafted baseline
            baseline = make_minimax_bot(depth=2)
            print(f"\n--- Benchmark: iter{i} vs handcrafted ---")
            tournament(neural_bot, baseline, n_games=10)

print("\n✓ Training complete!")