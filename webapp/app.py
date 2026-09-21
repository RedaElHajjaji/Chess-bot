# webapp/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import os, sys
sys.path.insert(0, os.path.abspath('..'))

from model.net import ChessNet, load_model
from engine.search import make_minimax_bot, make_neural_eval, random_move

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Load model (fallback to random if no checkpoint exists)
checkpoint = '../checkpoints/chess_net_v1.pth'
if os.path.exists(checkpoint):
    model = load_model(checkpoint)
    neural_eval = make_neural_eval(model)
    bot_fn = make_minimax_bot(depth=3, eval_fn=neural_eval)
    print("Neural bot loaded")
else:
    bot_fn = random_move
    print("No checkpoint found — using random bot")

game_board = chess.Board()   # shared game state

@app.route('/move', methods=['POST'])
def get_bot_move():
    data = request.json
    fen = data.get('fen')
    player_move_uci = data.get('player_move')

    game_board.set_fen(fen)

    # Apply player's move
    if player_move_uci:
        move = chess.Move.from_uci(player_move_uci)
        if move in game_board.legal_moves:
            game_board.push(move)
        else:
            return jsonify({'error': 'Illegal move'}), 400

    if game_board.is_game_over():
        return jsonify({
            'move': None, 'fen': game_board.fen(),
            'is_game_over': True, 'result': game_board.result()
        })

    # Get bot's reply
    bot_move = bot_fn(game_board)
    game_board.push(bot_move)

    return jsonify({
        'move': bot_move.uci(),
        'fen': game_board.fen(),
        'is_game_over': game_board.is_game_over(),
        'result': game_board.result() if game_board.is_game_over() else None
    })

@app.route('/legal_moves', methods=['POST'])
def get_legal_moves():
    data = request.json
    game_board.set_fen(data.get('fen'))
    sq = chess.parse_square(data.get('square'))
    moves = [m.uci() for m in game_board.legal_moves if m.from_square == sq]
    return jsonify({'legal_moves': moves})

@app.route('/reset', methods=['POST'])
def reset_game():
    game_board.reset()
    return jsonify({'fen': game_board.fen()})

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)





