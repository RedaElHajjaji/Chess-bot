# train_rl.py
import torch
import os
import sys
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
    print("Loaded existing checkpoint")
else:
    from model.net import ChessNet
    model = ChessNet().to(device)
    print("Starting fresh model")

# ── Training loop ──
N_ITERATIONS = 10

for i in range(1, N_ITERATIONS + 1):
    print(f"\n{'='*50}")
    print(f"ITERATION {i} / {N_ITERATIONS}")
    print(f"{'='*50}")

    model = training_iteration(
        model,
        iteration=i,
        games_per_iter=150,
        epochs=10,
        device=device,        # ← pass device through
        n_workers=4 
    )

    # Save latest after every iteration
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/chess_net_latest.pth')

    # Benchmark every 5 iterations
    if i % 5 == 0:
        neural_eval = make_neural_eval(model, device=device)
        neural_bot  = make_minimax_bot(depth=1, eval_fn=neural_eval)
        baseline    = make_minimax_bot(depth=1)
        print(f"\n--- Benchmark at iteration {i} ---")
        tournament(neural_bot, baseline, n_games=10)

print("\n✓ Training complete!")