# model/net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# All possible UCI moves in chess = 4672
# (8x8 from squares x 73 possible move types)
NUM_MOVES = 4672

# Build move index mapping
import chess

def build_move_index():
    """Map every possible UCI move to an index 0-4671."""
    move_to_idx = {}
    idx = 0
    for from_sq in range(64):
        for to_sq in range(64):
            if from_sq == to_sq:
                continue
            move = chess.Move(from_sq, to_sq)
            move_to_idx[move.uci()] = idx
            idx += 1
            # Promotions
            for promo in [chess.QUEEN, chess.ROOK,
                          chess.BISHOP, chess.KNIGHT]:
                promo_move = chess.Move(from_sq, to_sq, promotion=promo)
                move_to_idx[promo_move.uci()] = idx
                idx += 1
    return move_to_idx

MOVE_TO_IDX = build_move_index()
NUM_MOVES   = len(MOVE_TO_IDX)


class ResBlock(nn.Module):
    """Residual block — better at learning abstract patterns than plain CNN."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)   # skip connection
        return x


class ChessNet(nn.Module):
    """
    Dual-head network:
      - Value head:  position score -1 to +1
      - Policy head: move probabilities over all 4672 possible moves
    """
    def __init__(self, channels=128, n_res_blocks=6):
        super().__init__()

        # Input conv: 12 planes → channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(12, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        # Residual tower
        self.res_tower = nn.Sequential(
            *[ResBlock(channels) for _ in range(n_res_blocks)]
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Tanh()           # output: -1 to +1
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, NUM_MOVES)
            # No softmax here — use log_softmax in loss
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_tower(x)
        value  = self.value_head(x).squeeze(-1)    # (batch,)
        policy = self.policy_head(x)                # (batch, NUM_MOVES)
        return value, policy

    def predict(self, x):
        """Single position inference — returns (value, policy_probs)."""
        self.eval()
        with torch.no_grad():
            value, policy = self.forward(x)
            policy_probs = F.softmax(policy, dim=-1)
        return value, policy_probs


def load_model(path, device='cpu'):
    """Load checkpoint — handles both old single-head and new dual-head."""
    model = ChessNet()
    state = torch.load(path, map_location=device)

    try:
        model.load_state_dict(state)
        print(f"✓ Loaded dual-head model from {path}")
    except RuntimeError:
        # Old single-head checkpoint — start fresh
        print(f"⚠ Old checkpoint format detected — starting fresh dual-head model")
        model = ChessNet()

    return model.to(device)