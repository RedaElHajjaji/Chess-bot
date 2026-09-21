# train_rl.py  (run this from your project root)
from model.net import ChessNet, load_model
from self_play.loop import training_iteration
from engine.search import make_minimax_bot, make_neural_eval, tournament
import os

# Load existing checkpoint or start fresh
checkpoint = 'checkpoints/chess_net_v1.pth'
model = load_model(checkpoint) if os.path.exists(checkpoint) else ChessNet()

# Run training iterations
N_ITERATIONS = 20
for i in range(1, N_ITERATIONS + 1):
    model = training_iteration(model, iteration=i, games_per_iter=150)

    # Every 5 iterations, benchmark vs handcrafted bot
    if i % 5 == 0:
        neural_eval = make_neural_eval(model)
        neural_bot = make_minimax_bot(depth=1, eval_fn=neural_eval)
        baseline = make_minimax_bot(depth=1)
        print(f"\n--- Benchmark at iteration {i} ---")
        tournament(neural_bot, baseline, n_games=10)