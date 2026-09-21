from engine.search import random_move, make_minimax_bot, tournament

minimax_d3 = make_minimax_bot(depth=3)

print("Minimax-d3 vs Random:")
tournament(minimax_d3, random_move, n_games=20)   # expect 90%+

print("Random vs Minimax-d3:")
tournament(random_move, minimax_d3, n_games=20)   # expect <10%