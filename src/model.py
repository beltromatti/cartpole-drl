import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        """Initialize the Deep Q-Network (DQN) model.

        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Number of possible actions.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)  # First fully connected layer: state_size -> 256
        self.fc2 = nn.Linear(256, 128)         # Second fully connected layer: 256 -> 128
        self.fc3 = nn.Linear(128, action_size) # Output layer: 128 -> action_size

    def forward(self, x):
        """Define the forward pass of the network.

        Args:
            x (torch.Tensor): Input state tensor of shape [batch_size, state_size].

        Returns:
            torch.Tensor: Q-values for each action, shape [batch_size, action_size].
        """
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to first layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation to second layer
        x = self.fc3(x)             # Linear output: Q-values
        return x