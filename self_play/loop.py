# self_play/loop.py
import torch
import os
from tqdm import tqdm
from engine.search import play_game, make_minimax_bot, random_move

def generate_training_data(n_games=100, white_bot=None, black_bot=None):
    if white_bot is None:
        white_bot = make_minimax_bot(depth=2)
    if black_bot is None:
        black_bot = make_minimax_bot(depth=2)

    all_tensors = []
    all_labels = []

    for i in tqdm(range(n_games), desc="🎮 Generating games", unit="game"):
        history, result, result_str = play_game(white_bot, black_bot)

        for tensor, turn in history:
            label = result if turn else -result
            all_tensors.append(tensor)
            all_labels.append(torch.tensor(label, dtype=torch.float32))

        # Print a mini summary every 10 games
        if (i + 1) % 10 == 0:
            tqdm.write(f"  Game {i+1}: result={result_str}, positions so far={len(all_tensors)}")

    tensors = torch.stack(all_tensors)
    labels  = torch.stack(all_labels)

    os.makedirs('data', exist_ok=True)
    torch.save({'tensors': tensors, 'labels': labels}, 'data/training_data.pt')
    print(f"\n✓ Saved {len(tensors)} positions from {n_games} games")
    return tensors, labels


def self_play_game(model, depth=2):
    """
    Play one game where the neural bot plays both sides.
    Returns list of (tensor, label) training pairs.
    """
    from engine.search import make_minimax_bot, make_neural_eval
    from engine.board import get_result_value
    import chess

    neural_eval = make_neural_eval(model)
    bot = make_minimax_bot(depth=depth, eval_fn=neural_eval)

    board = chess.Board()
    history = []   # list of (tensor, turn)

    while not board.is_game_over() and len(board.move_stack) < 200:
        from engine.board import board_to_tensor
        tensor = board_to_tensor(board)
        history.append((tensor, board.turn))
        move = bot(board)
        board.push(move)

    result = get_result_value(board)

    training_pairs = []
    for tensor, turn in history:
        label = result if turn else -result
        training_pairs.append((tensor, torch.tensor(label, dtype=torch.float32)))

    return training_pairs



def training_iteration(model, iteration, games_per_iter=200, epochs=10):
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    print(f"\n=== Iteration {iteration} — Generating {games_per_iter} games ===")
    all_pairs = []

    for _ in tqdm(range(games_per_iter)):
        pairs = self_play_game(model, depth=2)
        all_pairs.extend(pairs)

    tensors = torch.stack([p[0] for p in all_pairs])
    labels  = torch.stack([p[1] for p in all_pairs])

    print(f"Training on {len(tensors)} positions...")
    dataset = TensorDataset(tensors, labels)
    loader  = DataLoader(dataset, batch_size=256, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for X, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch == epochs-1:
            print(f"  Final loss: {total_loss/len(loader):.4f}")

    path = f'checkpoints/chess_net_iter{iteration}.pth'
    torch.save(model.state_dict(), path)
    print(f"  Saved: {path}")
    return model