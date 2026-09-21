# model/net.py
import torch
import torch.nn as nn

class ChessNet(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.conv_block = nn.Sequential(
            # Input: (batch, 12, 8, 8)
            nn.Conv2d(12, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),

            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),

            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(),

            nn.Flatten()
            # Output: (batch, channels*2 * 8 * 8)
        )

        self.fc = nn.Sequential(
            nn.Linear(channels * 2 * 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()   # squash to [-1, +1]
        )

    def forward(self, x):
        return self.fc(self.conv_block(x)).squeeze(-1)

def load_model(path, device='cpu'):
    model = ChessNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.to(device)