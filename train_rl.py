# train_rl.py
import torch
import os
import sys
import shutil
import json
import subprocess
sys.path.insert(0, '/kaggle/working/Chess-bot')

from model.net import ChessNet, load_model
from self_play.loop import training_iteration_mcts
from engine.search import make_mcts_bot
from engine.mcts import make_mcts_bot as mcts_bot_factory

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
os.makedirs('/kaggle/working', exist_ok=True)

# ── Kaggle Dataset push ──
KAGGLE_USERNAME = "redaelhajjaji"
DATASET_ID      = f"{KAGGLE_USERNAME}/chess-bot-checkpoints"
DATASET_DIR     = '/kaggle/working/chess-checkpoints'

def push_to_dataset(iteration):
    os.makedirs(DATASET_DIR, exist_ok=True)
    shutil.copy(f'checkpoints/chess_net_iter{iteration}.pth',
                f'{DATASET_DIR}/chess_net_iter{iteration}.pth')
    shutil.copy('checkpoints/chess_net_latest.pth',
                f'{DATASET_DIR}/chess_net_latest.pth')

    meta = {
        "title": "chess-bot-checkpoints",
        "id": DATASET_ID,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(f'{DATASET_DIR}/dataset-metadata.json', 'w') as f:
        json.dump(meta, f)

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
        print(f"⚠ Dataset push failed: {result.stderr[:100]}")

# ── Training loop ──
N_ITERATIONS   = 20
N_SIMULATIONS  = 100   # MCTS simulations per move
GAMES_PER_ITER = 30    # fewer games but much richer data

for i in range(1, N_ITERATIONS + 1):
    print(f"\n{'='*50}")
    print(f"ITERATION {i} / {N_ITERATIONS}")
    print(f"{'='*50}")

    model = training_iteration_mcts(
        model,
        iteration=i,
        games_per_iter=GAMES_PER_ITER,
        epochs=3,
        device=device,
        n_simulations=N_SIMULATIONS
    )

    # Save everywhere
    torch.save(model.state_dict(), f'checkpoints/chess_net_iter{i}.pth')
    torch.save(model.state_dict(), 'checkpoints/chess_net_latest.pth')
    shutil.copy('checkpoints/chess_net_latest.pth',
                '/kaggle/working/chess_net_latest.pth')
    shutil.copy(f'checkpoints/chess_net_iter{i}.pth',
                f'/kaggle/working/chess_net_iter{i}.pth')
    push_to_dataset(i)
    print(f"✓ iter{i} saved everywhere — safe to stop anytime")

    # Benchmark every 5 iterations
    if i % 5 == 0:
        from engine.search import tournament, make_minimax_bot
        from engine.mcts import make_mcts_bot
        from model.net import load_model as lm

        print(f"\n--- Benchmark: iter{i} MCTS vs handcrafted minimax ---")
        model.cpu()
        mcts_bot = make_mcts_bot(model, device='cpu',
                                 n_simulations=50)
        baseline = make_minimax_bot(depth=2)
        tournament(mcts_bot, baseline, n_games=5)
        model.to(device)

print("\n✓ Training complete!")