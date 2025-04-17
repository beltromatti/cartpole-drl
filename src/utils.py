import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from .model import DQN

def select_action(state, epsilon, model, action_size):
    """Select an action using an epsilon-greedy policy.

    Args:
        state (torch.Tensor): Current state tensor of shape [1, state_size].
        epsilon (float): Epsilon value for exploration-exploitation trade-off.
        model (nn.Module): DQN model for predicting Q-values.
        action_size (int): Number of possible actions.

    Returns:
        int: Selected action index.
    """
    if random.random() > epsilon:  # Exploit: select action with highest Q-value
        with torch.no_grad():     # Disable gradient computation for inference
            q_values = model(state)
            return q_values.argmax().item()
    else:                         # Explore: select random action
        return random.randrange(action_size)

def optimize_model(model, target_model, memory, optimizer, batch_size, gamma):
    """Optimize the DQN model using a batch of transitions.

    Args:
        model (nn.Module): DQN model to optimize.
        target_model (nn.Module): Target DQN model for stable Q-value estimation.
        memory (ReplayMemory): Replay memory containing transitions.
        optimizer (optim.Optimizer): Optimizer for the model.
        batch_size (int): Number of transitions to sample.
        gamma (float): Discount factor for future rewards.
    """
    if len(memory) < batch_size:  # Skip if not enough transitions
        return

    transitions = memory.sample(batch_size)  # Sample a batch of transitions

    # Extract components from transitions
    batch_state = torch.cat([t[0] for t in transitions])          # [batch_size, state_size]
    batch_action = torch.LongTensor([t[1] for t in transitions]).view(-1, 1)  # [batch_size, 1]
    batch_reward = torch.FloatTensor([t[2] for t in transitions]) # [batch_size]
    batch_next_state = torch.cat([t[3] for t in transitions])     # [batch_size, state_size]
    batch_done = torch.FloatTensor([t[4] for t in transitions])   # [batch_size]

    # Compute current Q-values for the actions taken
    current_q_values = model(batch_state).gather(1, batch_action).squeeze(1)  # [batch_size]

    # Compute target Q-values using the target model
    with torch.no_grad():
        next_q_values = target_model(batch_next_state).max(1)[0]  # [batch_size]
        target_q_values = batch_reward + (gamma * next_q_values * (1 - batch_done))  # Bellman equation

    # Compute loss (mean squared error)
    loss = nn.MSELoss()(current_q_values, target_q_values)

    # Perform optimization
    optimizer.zero_grad()                     # Clear previous gradients
    loss.backward()                           # Compute gradients of loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients for stability
    optimizer.step()                          # Update model parameters

def visualize_episode(model, episode):
    """Visualize a single episode using the current model.

    Args:
        model (nn.Module): DQN model to use for action selection.
        episode (int): Episode number for display purposes.
        render_mode (str, optional): Rendering mode for the environment. Defaults to "human".
    """
    from .environment import create_environment
    env = create_environment(render_mode="human")  # Create environment for visualization
    print(f"Visualizing episode {episode + 1}")
    state, _ = env.reset()  # Reset environment
    state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor [1, state_size]
    done = False
    total_reward = 0
    timeout = 3000  # Maximum timesteps to prevent infinite loops
    timestep = 0

    while not done and timestep < timeout:
        with torch.no_grad():  # Disable gradients for visualization
            q_values = model(state)
            action = q_values.argmax().item()  # Select action with highest Q-value
        next_state, reward, done, _, _ = env.step(action)  # Perform action
        state = torch.FloatTensor(next_state).unsqueeze(0)  # Update state
        total_reward += reward
        timestep += 1

    print(f"Reward for visualized episode: {total_reward}")
    env.close()  # Close the environment and rendering window

def save_model(model, state_size, action_size, optimizer, episode, epsilon, rewards, filepath="data/checkpoints/model.pth"):
    """Save the model, optimizer, and training state to a file.

    Args:
        model (nn.Module): DQN model to save.
        state_size (int): Dimension of the state space.
        action_size (int): Number of possible actions.
        optimizer (optim.Optimizer): Optimizer state to save.
        episode (int): Current episode number.
        epsilon (float): Current epsilon value.
        rewards (list): List of episode rewards.
        filepath (str, optional): Path to save the checkpoint. Defaults to "data/checkpoints/model.pth".
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure directory exists
    model_data = {
        'model_state_dict': model.state_dict(),  # Save model weights
        'state_size': state_size,                # Save state size
        'action_size': action_size,              # Save action size
        'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state
        'episode': episode,                      # Save current episode
        'epsilon': epsilon,                      # Save epsilon value
        'rewards': rewards                       # Save reward history
    }
    torch.save(model_data, filepath)
    print(f"Model saved to: {filepath}")

def load_model(filepath="data/checkpoints/model.pth", learning_rate=0.0005):
    """Load a saved model and training state from a file.

    Args:
        filepath (str, optional): Path to the checkpoint file. Defaults to "data/checkpoints/model.pth".
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.0005.

    Returns:
        tuple: (model, state_size, action_size, optimizer, episode, epsilon, rewards)
    """
    if os.path.exists(filepath):
        model_data = torch.load(filepath, weights_only=False)  # Load checkpoint
        state_size = model_data['state_size']
        action_size = model_data['action_size']
        model = DQN(state_size, action_size)  # Initialize new DQN model
        model.load_state_dict(model_data['model_state_dict'])  # Load model weights
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Initialize optimizer
        optimizer.load_state_dict(model_data['optimizer_state_dict'])  # Load optimizer state
        episode = model_data['episode']  # Load episode number
        epsilon = model_data['epsilon']  # Load epsilon value
        rewards = model_data['rewards']  # Load reward history
        print(f"Model loaded from: {filepath}")
        return model, state_size, action_size, optimizer, episode, epsilon, rewards
    else:
        raise FileNotFoundError(f"No model found at: {filepath}")