# data/generate_stockfish.py
"""
Generate training data from Stockfish games.
Stockfish plays against itself at a set ELO level.
Every position is labeled with the game outcome.
This produces much higher quality training data
than weak bot vs weak bot self-play.
"""
import chess
import chess.engine
import torch
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.board import board_to_tensor
from tqdm import tqdm


def generate_stockfish_games(
        n_games=1000,
        stockfish_elo=1200,
        think_time=0.05,
        output_path='data/stockfish_data.pt',
        stockfish_path='/usr/games/stockfish'):
    """
    Generate training data from Stockfish self-play.

    Args:
        n_games:          Number of games to generate
        stockfish_elo:    ELO level for Stockfish (500-3000)
        think_time:       Seconds per move (lower = faster)
        output_path:      Where to save the data
        stockfish_path:   Path to stockfish binary
    """

    print(f"♟ Generating {n_games} Stockfish games at ELO {stockfish_elo}")
    print(f"  Think time: {think_time}s per move")
    print(f"  Output: {output_path}\n")

    # Launch Stockfish
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        print(f"❌ Could not launch Stockfish: {e}")
        print("   Make sure Stockfish is installed: sudo apt install stockfish")
        return None, None

    # Configure ELO limit
    engine.configure({
        "UCI_LimitStrength": True,
        "UCI_Elo": stockfish_elo
    })

    all_tensors     = []
    all_labels      = []
    results         = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
    positions_total = 0

    for game_idx in tqdm(range(n_games),
                         desc=f"🎮 Stockfish ELO {stockfish_elo}"):
        board      = chess.Board()
        positions  = []   # board tensors
        move_stack = []   # moves played

        # Play one full game
        while not board.is_game_over():
            try:
                result = engine.play(
                    board,
                    chess.engine.Limit(time=think_time)
                )
                positions.append(board_to_tensor(board))
                move_stack.append(result.move)
                board.push(result.move)
            except Exception:
                break

        # Get game outcome
        outcome = board.result()
        results[outcome] = results.get(outcome, 0) + 1

        if outcome == '1-0':   value =  1.0
        elif outcome == '0-1': value = -1.0
        else:                  value =  0.0

        # Label each position with outcome
        # from that side's perspective — FIXED version
        temp_board = chess.Board()
        for i, tensor in enumerate(positions):
            turn  = temp_board.turn
            label = value if turn == chess.WHITE else -value
            all_tensors.append(tensor)
            all_labels.append(
                torch.tensor(float(label), dtype=torch.float32)
            )
            if i < len(move_stack):
                temp_board.push(move_stack[i])

        positions_total += len(positions)

        # Print progress every 100 games
        if (game_idx + 1) % 100 == 0:
            tqdm.write(
                f"  Game {game_idx+1}/{n_games}: "
                f"W:{results['1-0']} "
                f"D:{results['1/2-1/2']} "
                f"L:{results['0-1']} | "
                f"Positions: {positions_total}"
            )

    engine.quit()

    # Save dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tensors = torch.stack(all_tensors)
    labels  = torch.stack(all_labels)

    torch.save({
        'tensors':       tensors,
        'labels':        labels,
        'n_games':       n_games,
        'stockfish_elo': stockfish_elo,
        'results':       results
    }, output_path)

    print(f"\n✓ Saved {len(tensors)} positions from {n_games} games")
    print(f"  Results: W:{results['1-0']} "
          f"D:{results['1/2-1/2']} "
          f"L:{results['0-1']}")
    print(f"  Avg positions per game: {len(tensors)//n_games}")
    print(f"  Saved to: {output_path}")

    return tensors, labels


if __name__ == '__main__':

    # Stage 1 — minimum ELO (1320 is Stockfish's lowest)
    generate_stockfish_games(
        n_games=500,
        stockfish_elo=1320,
        think_time=0.05,
        output_path='data/stockfish_1320.pt'
        stockfish_path='/usr/games/stockfish'
    )

    # Stage 2 — intermediate
    generate_stockfish_games(
        n_games=500,
        stockfish_elo=1500,
        think_time=0.05,
        output_path='data/stockfish_1500.pt'
    )

    # Stage 3 — stronger
    generate_stockfish_games(
        n_games=500,
        stockfish_elo=1800,
        think_time=0.05,
        output_path='data/stockfish_1800.pt'
    )

    print("\n✓ All datasets generated!")
    print("  Next: run python3 data/train_supervised.py")