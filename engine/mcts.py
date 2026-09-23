# engine/mcts.py
import chess
import torch
import torch.nn.functional as F
import math
import numpy as np
from engine.board import board_to_tensor
from model.net import MOVE_TO_IDX, NUM_MOVES

# UCB exploration constant — higher = more exploration
C_PUCT = 1.4


class MCTSNode:
    """A node in the Monte Carlo search tree."""

    def __init__(self, board, parent=None, prior=0.0):
        self.board    = board.copy()
        self.parent   = parent
        self.prior    = prior       # P(s,a) from policy head
        self.children = {}          # move → MCTSNode

        self.visit_count  = 0       # N(s,a)
        self.value_sum    = 0.0     # W(s,a)
        self.is_expanded  = False

    @property
    def mean_value(self):
        """Q(s,a) — average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits):
        """
        UCB1 score used to select which child to explore.
        Balances exploitation (mean_value) and exploration (prior / visits).
        """
        exploration = C_PUCT * self.prior * \
                      math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.mean_value + exploration

    def select_child(self):
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_move  = None
        best_child = None

        for move, child in self.children.items():
            score = child.ucb_score(self.visit_count)
            if score > best_score:
                best_score = score
                best_move  = move
                best_child = child

        return best_move, best_child

    def expand(self, policy_probs):
        """
        Expand this node — create children for all legal moves.
        policy_probs: probability distribution over all 4672 moves.
        """
        self.is_expanded = True
        legal_moves = list(self.board.legal_moves)

        for move in legal_moves:
            uci = move.uci()
            prior = policy_probs[MOVE_TO_IDX.get(uci, 0)].item()
            child_board = self.board.copy()
            child_board.push(move)
            self.children[move] = MCTSNode(
                child_board, parent=self, prior=prior
            )

    def backpropagate(self, value):
        """
        Update this node and all ancestors with the result.
        value is from White's perspective — flip for Black nodes.
        """
        self.visit_count += 1
        self.value_sum   += value

        if self.parent:
            # Flip value for parent (opposite color)
            self.parent.backpropagate(-value)

    def is_terminal(self):
        return self.board.is_game_over()

    def terminal_value(self):
        """Return the ground-truth value for terminal positions."""
        result = self.board.result()
        if result == "1-0": return  1.0
        if result == "0-1": return -1.0
        return 0.0


class MCTS:
    """
    Monte Carlo Tree Search with batched GPU inference.
    The key GPU optimization: collect BATCH_SIZE leaf nodes,
    evaluate them all at once on GPU, then backpropagate.
    """

    def __init__(self, model, device='cuda', n_simulations=200,
                 batch_size=32):
        self.model        = model
        self.device       = device
        self.n_simulations = n_simulations
        self.batch_size   = batch_size

    def get_move_probs(self, board, temperature=1.0):
        """
        Run MCTS from the given board position.
        Returns move probabilities based on visit counts.
        """
        root = MCTSNode(board)

        # Evaluate and expand root
        value, policy = self._evaluate_node(root)
        root.expand(policy[0])
        root.backpropagate(value[0].item())

        # Run simulations in batches
        sims_done = 0
        while sims_done < self.n_simulations:
            # Collect a batch of leaf nodes
            batch_nodes = []
            for _ in range(self.batch_size):
                if sims_done >= self.n_simulations:
                    break
                leaf = self._select(root)
                batch_nodes.append(leaf)
                sims_done += 1

            # Evaluate batch on GPU
            self._evaluate_and_expand_batch(batch_nodes)

        # Build move probability distribution from visit counts
        move_probs = {}
        total_visits = sum(
            child.visit_count for child in root.children.values()
        )

        for move, child in root.children.items():
            if temperature == 0:
                # Deterministic — always pick most visited
                move_probs[move] = child.visit_count
            else:
                move_probs[move] = child.visit_count ** (1.0 / temperature)

        # Normalize
        total = sum(move_probs.values())
        if total > 0:
            move_probs = {m: v/total for m, v in move_probs.items()}

        return move_probs

    def select_move(self, board, temperature=0.1):
        """Select the best move from the current position."""
        move_probs = self.get_move_probs(board, temperature)
        if not move_probs:
            # Fallback to random legal move
            import random
            return random.choice(list(board.legal_moves))

        moves = list(move_probs.keys())
        probs = list(move_probs.values())

        if temperature == 0:
            return moves[probs.index(max(probs))]
        else:
            # Sample from distribution
            probs_array = np.array(probs)
            probs_array = probs_array / probs_array.sum()
            return moves[np.random.choice(len(moves), p=probs_array)]

    def _select(self, root):
        """Walk tree selecting best UCB child until reaching a leaf."""
        node = root
        while node.is_expanded and not node.is_terminal():
            _, node = node.select_child()
            if node is None:
                break
        return node

    def _evaluate_node(self, node):
        """Evaluate a single node on GPU."""
        tensor = board_to_tensor(node.board).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            value, policy = self.model(tensor)
            policy = F.softmax(policy, dim=-1)
        return value, policy

    def _evaluate_and_expand_batch(self, nodes):
        """
        The key GPU optimization:
        Evaluate a batch of nodes all at once on GPU.
        """
        # Filter out terminal nodes
        non_terminal = [n for n in nodes if not n.is_terminal()]
        terminal     = [n for n in nodes if n.is_terminal()]

        # Handle terminal nodes directly
        for node in terminal:
            node.backpropagate(node.terminal_value())

        if not non_terminal:
            return

        # Stack all board tensors into one batch
        tensors = torch.stack([
            board_to_tensor(n.board) for n in non_terminal
        ]).to(self.device)

        # Single GPU forward pass for entire batch
        self.model.eval()
        with torch.no_grad():
            values, policies = self.model(tensors)
            policies = F.softmax(policies, dim=-1)

        # Expand and backpropagate each node
        for i, node in enumerate(non_terminal):
            if not node.is_expanded:
                node.expand(policies[i])
            node.backpropagate(values[i].item())


def make_mcts_bot(model, device='cuda', n_simulations=200,
                  temperature=0.1):
    """
    Factory: returns a bot function that uses MCTS.
    Compatible with play_game() and tournament().
    """
    mcts = MCTS(model, device=device, n_simulations=n_simulations)

    def bot(board):
        return mcts.select_move(board, temperature=temperature)

    return bot