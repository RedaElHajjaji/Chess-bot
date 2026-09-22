# self_play/loop.py
import torch
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
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
            tqdm.write(f"  Game {i+1}: result={result_str}, "
                      f"positions so far={len(all_tensors)}")

    tensors = torch.stack(all_tensors)
    labels  = torch.stack(all_labels)

    os.makedirs('data', exist_ok=True)
    torch.save({'tensors': tensors, 'labels': labels}, 'data/training_data.pt')
    print(f"\n✓ Saved {len(tensors)} positions from {n_games} games")
    return tensors, labels


def self_play_game(cpu_model, depth=2):
    """Single game — always runs on CPU."""
    from engine.search import make_minimax_bot, make_neural_eval
    from engine.board import board_to_tensor, get_result_value
    import chess

    neural_eval = make_neural_eval(cpu_model, device='cpu')
    bot = make_minimax_bot(depth=depth, eval_fn=neural_eval)

    board = chess.Board()
    history = []

    while not board.is_game_over() and len(board.move_stack) < 80:
        tensor = board_to_tensor(board)
        history.append((tensor, board.turn))
        move = bot(board)
        board.push(move)

    result = get_result_value(board)
    training_pairs = []
    for tensor, turn in history:
        label = result if turn else -result
        training_pairs.append((
            tensor,
            torch.tensor(float(label), dtype=torch.float32)
        ))
    return training_pairs


def generate_threaded(cpu_model, n_games=50, depth=2, n_workers=4):
    """
    Run n_games in parallel using threads.
    Threads don't deadlock like multiprocessing on Kaggle.
    """
    from engine.search import make_minimax_bot, make_neural_eval
    from engine.board import board_to_tensor, get_result_value
    import chess

    neural_eval = make_neural_eval(cpu_model, device='cpu')
    bot = make_minimax_bot(depth=depth, eval_fn=neural_eval)

    def play_one_game(_):
        board = chess.Board()
        history = []

        while not board.is_game_over() and len(board.move_stack) < 80:
            tensor = board_to_tensor(board)
            history.append((tensor, board.turn))
            move = bot(board)
            board.push(move)

        result = get_result_value(board)
        pairs = []
        for tensor, turn in history:
            label = result if turn else -result
            pairs.append((
                tensor,
                torch.tensor(float(label), dtype=torch.float32)
            ))
        return pairs

    all_pairs = []
    completed = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(play_one_game, i) for i in range(n_games)]

        for future in tqdm(futures, total=n_games,
                           desc="🎮 Generating games (threaded)"):
            try:
                pairs = future.result()
                all_pairs.extend(pairs)
                completed += 1
                if completed % 10 == 0:
                    tqdm.write(f"  {completed}/{n_games} games done, "
                              f"{len(all_pairs)} positions collected")
            except Exception as e:
                tqdm.write(f"  ⚠ Game failed: {e} — skipping")
                continue

    return all_pairs


def training_iteration(model, iteration, games_per_iter=50,
                       epochs=3, device='cpu', n_workers=4):
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from model.net import ChessNet

    print(f"\n=== Iteration {iteration} — Generating {games_per_iter} games ===")
    os.makedirs('checkpoints', exist_ok=True)

    # Save current weights to temp file
    tmp_path = 'checkpoints/tmp_worker.pth'
    torch.save(model.state_dict(), tmp_path)

    # Load separate CPU copy for game generation
    cpu_model = ChessNet()
    cpu_model.load_state_dict(torch.load(tmp_path, map_location='cpu'))
    cpu_model.eval()

    # Generate games using threads — parallel, no deadlocks
    all_pairs = generate_threaded(cpu_model,
                                  n_games=games_per_iter,
                                  depth=2,
                                  n_workers=n_workers)

    if len(all_pairs) == 0:
        print("⚠ No pairs generated — skipping training this iteration")
        return model

    # Move data to training device
    tensors = torch.stack([
        p[0] if isinstance(p[0], torch.Tensor)
        else torch.tensor(p[0])
        for p in all_pairs
    ]).to(device)
    labels = torch.stack([
        p[1].detach().clone().float()
        if isinstance(p[1], torch.Tensor)
        else torch.tensor(float(p[1]), dtype=torch.float32)
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

        avg_loss = total_loss / len(loader)
        tqdm.write(f"  Epoch {epoch+1}/{epochs}  loss: {avg_loss:.4f}")

        # Early stop if loss is good enough
        if avg_loss < 0.005:
            tqdm.write(f"  ⚡ Early stop — loss below threshold")
            break

    path = f'checkpoints/chess_net_iter{iteration}.pth'
    torch.save(model.state_dict(), path)
    torch.save(model.state_dict(), 'checkpoints/chess_net_latest.pth')
    print(f"✓ Saved: {path}")
    return model