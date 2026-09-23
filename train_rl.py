# train_rl.py
import torch
import os
import sys
import shutil
import json
import subprocess
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
os.makedirs('/kaggle/working/chess-checkpoints', exist_ok=True)

# ── Kaggle Dataset push function ──
KAGGLE_USERNAME = "redaelhajjaji"   # ← your Kaggle username
DATASET_ID = f"{KAGGLE_USERNAME}/chess-bot-checkpoints"
DATASET_DIR = '/kaggle/working/chess-checkpoints'

def push_to_dataset(iteration):
    """Push checkpoint to persistent Kaggle Dataset after every iteration."""
    # Copy checkpoints to dataset folder
    shutil.copy(f'checkpoints/chess_net_iter{iteration}.pth',
                f'{DATASET_DIR}/chess_net_iter{iteration}.pth')
    shutil.copy('checkpoints/chess_net_latest.pth',
                f'{DATASET_DIR}/chess_net_latest.pth')

    # Write metadata
    meta = {
        "title": "chess-bot-checkpoints",
        "id": DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(f'{DATASET_DIR}/dataset-metadata.json', 'w') as f:
        json.dump(meta, f)

    # Push new version
    result = subprocess.run(
        ['kaggle', 'datasets', 'version',
         '-p', DATASET_DIR,
         '-m', f'iter{iteration}',
         '--dir-mode', 'zip'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"✓ Pushed iter{iteration} to Kaggle Dataset")
    else:
        print(f"⚠ Dataset push failed: {result.stderr}")
        print(f"  (checkpoint still saved to /kaggle/working/ as backup)")

# ── Training loop ──
N_ITERATIONS = 20

for i in range(1, N_ITERATIONS + 1):
    print(f"\n{'='*50}")
    print(f"ITERATION {i} / {N_ITERATIONS}")
    print(f"{'='*50}")

    model = training_iteration(
        model,
        iteration=i,
        games_per_iter=50,
        epochs=3,            # reduced to prevent memorization
        device=device,
        n_workers=4
    )

    # Save locally
    torch.save(model.state_dict(), f'checkpoints/chess_net_iter{i}.pth')
    torch.save(model.state_dict(), 'checkpoints/chess_net_latest.pth')

    # Save to /kaggle/working/ output
    shutil.copy('checkpoints/chess_net_latest.pth',
                '/kaggle/working/chess_net_latest.pth')
    shutil.copy(f'checkpoints/chess_net_iter{i}.pth',
                f'/kaggle/working/chess_net_iter{i}.pth')

    # Push to persistent Kaggle Dataset
    push_to_dataset(i)

    print(f"✓ iter{i} saved everywhere — safe to stop anytime")

    # Benchmark every 5 iterations
    # Benchmark every 5 iterations — depth 1 to match training
    # Benchmark every 5 iterations — depth 1 to match training
    if i % 5 == 0:
        from model.net import load_model as lm

        # Move current model to CPU for benchmark
        model.cpu()
        neural_eval = make_neural_eval(model, device='cpu')
        neural_bot  = make_minimax_bot(depth=1, eval_fn=neural_eval)

        iter1_path = 'checkpoints/chess_net_iter1.pth'
        if os.path.exists(iter1_path):
            early_model = lm(iter1_path, device='cpu')
            early_eval  = make_neural_eval(early_model, device='cpu')
            early_bot   = make_minimax_bot(depth=1, eval_fn=early_eval)
            print(f"\n--- Benchmark: iter{i} vs iter1 (depth 1) ---")
            tournament(neural_bot, early_bot, n_games=10)
        else:
            baseline = make_minimax_bot(depth=1)
            print(f"\n--- Benchmark: iter{i} vs handcrafted (depth 1) ---")
            tournament(neural_bot, baseline, n_games=10)

        # Move model back to GPU for next training iteration
        model.to(device)

print("\n✓ Training complete!")