# webapp/app.py
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import chess
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

from model.net import ChessNet, load_model
from engine.search import (make_personality_bot, make_personality_eval,
                            make_minimax_bot, tournament)
from engine.board import board_to_tensor

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# ── Load model ──
CHECKPOINT = '../checkpoints/chess_net_latest.pth'
device = 'cpu'   # web app always uses CPU for fast response

if os.path.exists(CHECKPOINT):
    model = load_model(CHECKPOINT, device=device)
    model.eval()
    print("✓ Neural model loaded")
else:
    model = None
    print("⚠ No checkpoint found — using handcrafted eval")

# ── Pre-build all personality bots ──
def build_bots(model, device='cpu'):
    if model is None:
        return {p: make_minimax_bot(depth=2)
                for p in ['balanced','greedy','defensive','aggressive']}
    return {
        'balanced':   make_personality_bot(model, 'balanced',   device, depth=2),
        'greedy':     make_personality_bot(model, 'greedy',     device, depth=2),
        'defensive':  make_personality_bot(model, 'defensive',  device, depth=2),
        'aggressive': make_personality_bot(model, 'aggressive', device, depth=2),
    }

bots = build_bots(model, device)

# ── Game state ──
game_board = chess.Board()

# ── Personality ELO estimates ──
PERSONALITY_INFO = {
    'balanced':   {'name': 'Balanced',   'emoji': '⚖️',  'elo': '~900'},
    'greedy':     {'name': 'Greedy',     'emoji': '💰',  'elo': '~800'},
    'defensive':  {'name': 'Defensive',  'emoji': '🛡️',  'elo': '~850'},
    'aggressive': {'name': 'Aggressive', 'emoji': '⚔️',  'elo': '~950'},
}

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/move', methods=['POST'])
def get_bot_move():
    data        = request.json
    fen         = data.get('fen')
    player_move = data.get('player_move')
    personality = data.get('personality', 'balanced')

    game_board.set_fen(fen)

    # Apply player move
    if player_move:
        move = chess.Move.from_uci(player_move)
        if move in game_board.legal_moves:
            game_board.push(move)
        else:
            return jsonify({'error': 'Illegal move'}), 400

    if game_board.is_game_over():
        return jsonify({
            'move': None,
            'fen': game_board.fen(),
            'is_game_over': True,
            'result': game_board.result(),
            'reason': _game_over_reason(game_board)
        })

    # Get bot move
    bot_fn   = bots.get(personality, bots['balanced'])
    bot_move = bot_fn(game_board)
    game_board.push(bot_move)

    # Check if king is in check after bot move
    in_check = game_board.is_check()
    king_sq  = None
    if in_check:
        king_sq = chess.square_name(game_board.king(game_board.turn))

    return jsonify({
        'move':        bot_move.uci(),
        'fen':         game_board.fen(),
        'is_game_over': game_board.is_game_over(),
        'result':      game_board.result() if game_board.is_game_over() else None,
        'reason':      _game_over_reason(game_board) if game_board.is_game_over() else None,
        'in_check':    in_check,
        'king_square': king_sq
    })

@app.route('/legal_moves', methods=['POST'])
def get_legal_moves():
    data = request.json
    game_board.set_fen(data.get('fen'))
    sq    = chess.parse_square(data.get('square'))
    moves = [m.uci() for m in game_board.legal_moves
             if m.from_square == sq]
    return jsonify({'legal_moves': moves})

@app.route('/reset', methods=['POST'])
def reset_game():
    game_board.reset()
    return jsonify({'fen': game_board.fen()})

@app.route('/personalities', methods=['GET'])
def get_personalities():
    return jsonify(PERSONALITY_INFO)

def _game_over_reason(board):
    if board.is_checkmate():        return 'Checkmate'
    if board.is_stalemate():        return 'Stalemate'
    if board.is_insufficient_material(): return 'Insufficient material'
    if board.is_seventyfive_moves(): return '75-move rule'
    if board.is_fivefold_repetition(): return 'Fivefold repetition'
    return 'Draw'

if __name__ == '__main__':
    app.run(debug=True, port=5000)