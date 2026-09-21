# test_phase4.py
from self_play.loop import generate_training_data
from model.train import train
from model.net import load_model
from engine.search import make_minimax_bot, make_neural_eval, tournament

# Step 1: Generate 300 games of minimax-d2 vs minimax-d2
generate_training_data(n_games=100)

# Step 2: Train the network
train(epochs=15)

# Step 3: Load and test
model = load_model('checkpoints/chess_net_v1.pth')
neural_eval = make_neural_eval(model)
neural_bot = make_minimax_bot(depth=2, eval_fn=neural_eval)
handcrafted_bot = make_minimax_bot(depth=2)

print("Neural-d3 vs Handcrafted-d3:")
tournament(neural_bot, handcrafted_bot, n_games=10)