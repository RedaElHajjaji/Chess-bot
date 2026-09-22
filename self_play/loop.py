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

        if (i + 1) % 10 == 0:
            tqdm.write(f"  Game {i+1}: result={result_str}, positions so far={len(all_tensors)}")

    tensors = torch.stack(all_tensors)
    labels  = torch.stack(all_labels)

    os.makedirs('data', exist_ok=True)
    torch.save({'tensors': tensors, 'labels': labels}, 'data/training_data.pt')
    print(f"\n✓ Saved {len(tensors)} positions from {n_games} games")
    return tensors, labels


def self_play_game(model, depth=2, device='cpu'):
    from engine.search import make_minimax_bot, make_neural_eval
    from engine.board import board_to_tensor, get_result_value
    import chess

    neural_eval = make_neural_eval(model, device=device)
    bot = make_minimax_bot(depth=depth, eval_fn=neural_eval)

    board = chess.Board()
    history = []

    while not board.is_game_over() and len(board.move_stack) < 200:
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


def generate_sequential(model, n_games=50, depth=2, device='cpu'):
    """Simple sequential game generation — no multiprocessing."""
    all_pairs = []

    for i in tqdm(range(n_games), desc="🎮 Generating games"):
        pairs = self_play_game(model, depth=depth, device='cpu')
        all_pairs.extend(pairs)
        if (i + 1) % 10 == 0:
            tqdm.write(f"  {i+1}/{n_games} games done, "
                      f"{len(all_pairs)} positions collected")

    return all_pairs


def training_iteration(model, iteration, games_per_iter=50,
                       epochs=5, device='cpu', n_workers=4):
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    print(f"\n=== Iteration {iteration} — Generating {games_per_iter} games ===")

    os.makedirs('checkpoints', exist_ok=True)

    # Sequential — no multiprocessing, no deadlocks
    all_pairs = generate_sequential(model, n_games=games_per_iter,
                                    depth=2, device=device)

    tensors = torch.stack([
        p[0] if isinstance(p[0], torch.Tensor) else torch.tensor(p[0])
        for p in all_pairs
    ]).to(device)
    labels = torch.stack([
        torch.tensor(p[1], dtype=torch.float32)
        for p in all_pairs
    ]).to(device)

    print(f"📦 {len(tensors)} positions → training on {device}...")
    dataset = TensorDataset(tensors, labels)
    loader  = DataLoader(dataset, batch_size=256, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn   = nn.MSELoss()
    model.train()

    for epoch in tqdm(range(epochs), desc="🧠 Training", unit="epoch"):
        total_loss = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        tqdm.write(f"  Epoch {epoch+1}/{epochs}  "
                   f"loss: {total_loss/len(loader):.4f}")

    path = f'checkpoints/chess_net_iter{iteration}.pth'
    torch.save(model.state_dict(), path)
    torch.save(model.state_dict(), 'checkpoints/chess_net_latest.pth')
    print(f"✓ Saved: {path}")
    return model