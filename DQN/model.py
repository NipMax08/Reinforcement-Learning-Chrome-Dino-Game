import torch
import torch.nn as nn

# Define the Deep Q-Network (DQN) model
class DQN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
         # Initialize the parent class (nn.Module)
        super().__init__(*args, **kwargs)

        # Define the convolutional layers to process image-like inputs (e.g., frames from the environment)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Define the fully connected (dense) layers for final decision-making
        self.fc = nn.Sequential(
            nn.Linear(3072, 256), nn.ReLU(), nn.Linear(256, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define the forward pass of the model
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
