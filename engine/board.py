# engine/board.py
import chess
import torch
import numpy as np

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Returns a (12, 8, 8) float tensor.
    Planes 0-5:  White PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
    Planes 6-11: Black PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
    """
    planes = torch.zeros(12, 8, 8, dtype=torch.float32)

    for i, piece_type in enumerate(PIECE_TYPES):
        for sq in board.pieces(piece_type, chess.WHITE):
            row, col = divmod(sq, 8)
            planes[i][row][col] = 1.0
        for sq in board.pieces(piece_type, chess.BLACK):
            row, col = divmod(sq, 8)
            planes[i + 6][row][col] = 1.0

    return planes

def get_result_value(board: chess.Board) -> float:
    """Returns +1 for White win, -1 for Black win, 0 for draw."""
    result = board.result()
    if result == "1-0": return 1.0
    if result == "0-1": return -1.0
    return 0.0