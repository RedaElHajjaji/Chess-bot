# engine/search.py
import chess
import random
from tqdm import tqdm

def random_move(board: chess.Board) -> chess.Move:
    """Pick a random legal move."""
    return random.choice(list(board.legal_moves))


# Add to engine/search.py

def play_game(white_fn, black_fn, max_moves=200):
    """
    Play a full game between two move functions.
    white_fn(board) -> Move
    black_fn(board) -> Move
    Returns: list of (board_tensor, color_to_move, result)
    """
    from engine.board import board_to_tensor, get_result_value

    board = chess.Board()
    history = []

    while not board.is_game_over() and len(board.move_stack) < max_moves:
        tensor = board_to_tensor(board)
        turn = board.turn  # chess.WHITE or chess.BLACK

        if turn == chess.WHITE:
            move = white_fn(board)
        else:
            move = black_fn(board)

        history.append((tensor, turn))
        board.push(move)

    result = get_result_value(board)
    return history, result, board.result()


# Add to engine/search.py

def tournament(bot_a, bot_b, n_games=50):
    """
    Play n_games between bot_a (White) and bot_b (Black).
    Returns win rate for bot_a.
    """
    wins = draws = losses = 0

    for i in tqdm(range(n_games), desc="🏆 Tournament", unit="game"):
        _, _, result = play_game(bot_a, bot_b)
        if result == "1-0": wins += 1
        elif result == "0-1": losses += 1
        else: draws += 1

        # Live score update every game
        tqdm.write(f"  Game {i+1}: {result} | W:{wins} D:{draws} L:{losses}")

    win_rate = (wins + 0.5 * draws) / n_games
    print(f"\n✓ Final — W:{wins} D:{draws} L:{losses} | Win rate: {win_rate:.2%}")
    return win_rate



# Add to engine/search.py

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0
}

# Piece-square tables: reward pieces on good squares
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

def evaluate_handcrafted(board: chess.Board) -> float:
    if board.is_checkmate():
        return -99999 if board.turn == chess.WHITE else 99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value

    # Add pawn position bonus
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        score += PAWN_TABLE[sq]
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        score -= PAWN_TABLE[63 - sq]   # mirror for Black

    return score



def minimax(board, depth, alpha, beta, maximizing, eval_fn):
    if depth == 0 or board.is_game_over():
        return eval_fn(board)

    if maximizing:
        best = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth-1, alpha, beta, False, eval_fn)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break   # beta cut-off (prune)
        return best
    else:
        best = float('inf')
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth-1, alpha, beta, True, eval_fn)
            board.pop()
            best = min(best, score)
            beta = min(beta, score)
            if beta <= alpha:
                break   # alpha cut-off (prune)
        return best

def best_move_minimax(board, depth=3, eval_fn=None):
    if eval_fn is None:
        eval_fn = evaluate_handcrafted

    best_move = None
    best_score = -float('inf') if board.turn == chess.WHITE else float('inf')
    maximizing = (board.turn == chess.WHITE)

    for move in board.legal_moves:
        board.push(move)
        score = minimax(board, depth-1, -float('inf'), float('inf'),
                        not maximizing, eval_fn)
        board.pop()

        if (maximizing and score > best_score) or \
           (not maximizing and score < best_score):
            best_score = score
            best_move = move

    return best_move


def order_moves(board: chess.Board):
    """Return moves sorted: captures first, then checks, then quiet."""
    def move_priority(move):
        score = 0
        if board.is_capture(move):
            # MVV-LVA: prefer capturing high-value pieces with low-value pieces
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            if victim and attacker:
                score += PIECE_VALUES.get(victim.piece_type, 0) * 10
                score -= PIECE_VALUES.get(attacker.piece_type, 0)
        board.push(move)
        if board.is_check():
            score += 50    # bonus for giving check
        board.pop()
        return -score   # negative because sorted() is ascending

    return sorted(board.legal_moves, key=move_priority)


def make_minimax_bot(depth=3, eval_fn=None):
    """Factory: returns a bot function with fixed depth and eval."""
    def bot(board):
        return best_move_minimax(board, depth=depth, eval_fn=eval_fn)
    return bot

# Usage:
minimax_bot_d3 = make_minimax_bot(depth=3)
minimax_bot_d4 = make_minimax_bot(depth=4)   # stronger but slowe


# Add to engine/search.py
import torch
from engine.board import board_to_tensor

def make_neural_eval(model):
    """Returns an eval function that uses the neural network."""
    model.eval()
    def neural_eval(board):
        if board.is_checkmate():
            return -99999 if board.turn == True else 99999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        with torch.no_grad():
            tensor = board_to_tensor(board).unsqueeze(0)  # add batch dim
            score = model(tensor).item()
        return score * 10000   # scale to match material values
    return neural_eval

# Usage:
# from model.net import load_model
# model = load_model('checkpoints/chess_net_v1.pth')
# neural_eval = make_neural_eval(model)
# neural_bot = make_minimax_bot(depth=3, eval_fn=neural_eval)