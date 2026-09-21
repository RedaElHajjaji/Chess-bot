# Save as: test_phase2.py
from engine.search import random_move, play_game, tournament
from engine.board import board_to_tensor

# Test a single game
history, result, result_str = play_game(random_move, random_move)
print(f"Game: {len(history)} positions, result: {result_str}")
print(f"Tensor shape: {history[0][0].shape}")   # should be (12, 8, 8)

# Random vs random tournament (should be ~50% win rate)
print("Random vs Random:")
tournament(random_move, random_move, n_games=20)