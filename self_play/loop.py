# self_play/loop.py
import torch
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from engine.search import play_game, make_minimax_bot, random_move
from model.net import MOVE_TO_IDX, NUM_MOVES


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
    torch.save({'tensors': tensors, 'labels': labels},
               'data/training_data.pt')
    print(f"\n✓ Saved {len(tensors)} positions from {n_games} games")
    return tensors, labels


def self_play_game_mcts(model, device='cuda', n_simulations=100):
    """
    Play one game using MCTS.
    Returns list of (board_tensor, policy_target, value_target).
    """
    from engine.mcts import MCTS
    from engine.board import board_to_tensor, get_result_value
    import chess

    mcts = MCTS(model, device=device, n_simulations=n_simulations)
    board = chess.Board()
    history = []   # (tensor, move_probs, turn)

    move_count = 0
    while not board.is_game_over() and move_count < 150:
        # Get move probabilities from MCTS
        temp = 1.0 if move_count < 20 else 0.1
        move_probs = mcts.get_move_probs(board, temperature=temp)

        # Record position
        tensor = board_to_tensor(board)

        # Build policy target vector
        policy_target = torch.zeros(NUM_MOVES)
        for move, prob in move_probs.items():
            idx = MOVE_TO_IDX.get(move.uci(), 0)
            policy_target[idx] = prob

        history.append((tensor, policy_target, board.turn))

        # Select and play move
        moves = list(move_probs.keys())
        probs = list(move_probs.values())
        import numpy as np
        probs_arr = np.array(probs)
        probs_arr = probs_arr / probs_arr.sum()
        move = moves[np.random.choice(len(moves), p=probs_arr)]
        board.push(move)
        move_count += 1

    result = get_result_value(board)

    # Label each position with outcome from that side's perspective
    training_data = []
    for tensor, policy_target, turn in history:
        value_target = result if turn else -result
        training_data.append((
            tensor,
            policy_target,
            torch.tensor(float(value_target), dtype=torch.float32)
        ))

    return training_data


def generate_mcts_games(model, n_games=50, device='cuda',
                        n_simulations=100):
    """Generate games using MCTS — runs sequentially since GPU is shared."""
    all_data = []

    for i in tqdm(range(n_games), desc="🎮 MCTS self-play"):
        game_data = self_play_game_mcts(model, device=device,
                                        n_simulations=n_simulations)
        all_data.extend(game_data)
        if (i + 1) % 5 == 0:
            tqdm.write(f"  {i+1}/{n_games} games done, "
                      f"{len(all_data)} positions collected")

    return all_data


def training_iteration_mcts(model, iteration, games_per_iter=30,
                             epochs=3, device='cuda',
                             n_simulations=100):
    """
    One training iteration using MCTS self-play.
    GPU is used for:
      - MCTS batch inference (game generation)
      - Neural network training
    """
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader

    print(f"\n=== Iteration {iteration} — "
          f"Generating {games_per_iter} MCTS games ===")
    os.makedirs('checkpoints', exist_ok=True)

    model.to(device)
    model.eval()

    # Generate games — GPU used here for batch inference
    all_data = generate_mcts_games(model, n_games=games_per_iter,
                                   device=device,
                                   n_simulations=n_simulations)

    if len(all_data) == 0:
        print("⚠ No data generated — skipping")
        return model

    # Unpack training data
    tensors  = torch.stack([d[0] for d in all_data]).to(device)
    policies = torch.stack([d[1] for d in all_data]).to(device)
    values   = torch.stack([d[2] for d in all_data]).to(device)

    print(f"📦 {len(tensors)} positions → training on {device}...")

    dataset = TensorDataset(tensors, policies, values)
    loader  = DataLoader(dataset, batch_size=512, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=5, gamma=0.5)
    model.train()

    for epoch in tqdm(range(epochs), desc="🧠 Training", unit="epoch"):
        total_value_loss  = 0
        total_policy_loss = 0

        for X, pol_target, val_target in loader:
            X          = X.to(device)
            pol_target = pol_target.to(device)
            val_target = val_target.to(device)

            optimizer.zero_grad(set_to_none=True)

            value, policy = model(X)

            # Value loss — MSE
            value_loss = nn.MSELoss()(value, val_target)

            # Policy loss — cross entropy
            policy_loss = -torch.mean(
                torch.sum(
                    pol_target * F.log_softmax(policy, dim=-1),
                    dim=-1
                )
            )

            # Combined loss
            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()

            total_value_loss  += value_loss.item()
            total_policy_loss += policy_loss.item()

        scheduler.step()
        avg_v = total_value_loss  / len(loader)
        avg_p = total_policy_loss / len(loader)
        tqdm.write(f"  Epoch {epoch+1}/{epochs}  "
                   f"value_loss: {avg_v:.4f}  policy_loss: {avg_p:.4f}")

        if avg_v < 0.005 and avg_p < 0.1:
            tqdm.write("  ⚡ Early stop")
            break

    path = f'checkpoints/chess_net_iter{iteration}.pth'
    torch.save(model.state_dict(), path)
    torch.save(model.state_dict(), 'checkpoints/chess_net_latest.pth')
    print(f"✓ Saved: {path}")
    return model