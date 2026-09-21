# Save as: test_setup.py
import chess
import random

board = chess.Board()
move_count = 0

while not board.is_game_over() and move_count < 1000:
    move = random.choice(list(board.legal_moves))
    board.push(move)
    move_count += 1

print(f"Game ran for {move_count} moves")
print(f"Game over: {board.is_game_over()}")
print(f"Result: {board.result()}")
print(board)   # prints ASCII board