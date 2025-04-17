# Import necessary libraries for the DQN project
import gymnasium as gym         # Import Gymnasium for creating and managing the reinforcement learning environment
import torch                    # Import PyTorch for neural network operations and tensor computations
import torch.nn as nn           # Import PyTorch's neural network module for defining network architectures
import torch.optim as optim     # Import PyTorch's optimization module for optimizers like Adam
import numpy as np              # Import NumPy for array operations and mathematical computations
import random                   # Import random for generating random numbers (used in exploration)
from collections import deque   # Import deque for implementing the replay memory with fixed capacity
import matplotlib.pyplot as plt # Import Matplotlib for visualizing training results (reward plots)
import os                       # Import os for file operations (e.g., saving/loading model checkpoints)

# Define global training parameters for the DQN agent
EPISODES = 1000                 # Set total number of training episodes to 1000
BATCH_SIZE = 64                 # Set batch size for training to 64 (increased for learning stability)
GAMMA = 0.99                    # Set discount factor for future rewards to 0.99
EPSILON_START = 1.0             # Set initial epsilon for epsilon-greedy policy to 1.0 (full exploration)
EPSILON_END = 0.02              # Set final epsilon to 0.02 (minimal exploration for residual randomness)
EPSILON_DECAY = 0.995           # Set epsilon decay rate per episode to 0.995
MEMORY_SIZE = 20000             # Set replay memory capacity to 20,000 transitions (increased for more experience)
LEARNING_RATE = 0.0005          # Set learning rate for the optimizer to 0.0005 (low for stable convergence)
TARGET_UPDATE = 10              # Set frequency (in episodes) for updating the target network to 10
VISUALIZE_EVERY = 50            # Set frequency for visualizing an episode to every 50 episodes
SAVE_CHECKPOINT_EVERY = 50      # Set frequency for saving model checkpoints to every 50 episodes

# Define the DQN class for the neural network
class DQN(nn.Module):           # Inherit from PyTorch's nn.Module to create a neural network class
    def __init__(self, state_size, action_size): # Initialize the DQN with state and action space sizes
        super(DQN, self).__init__()            # Call the parent class (nn.Module) initializer
        self.fc1 = nn.Linear(state_size, 256)  # Define first fully connected layer: input = state_size, output = 256 neurons
        self.fc2 = nn.Linear(256, 128)         # Define second fully connected layer: input = 256, output = 128 neurons
        self.fc3 = nn.Linear(128, action_size) # Define output layer: input = 128, output = number of actions

    def forward(self, x):           # Define the forward pass of the network
        x = torch.relu(self.fc1(x)) # Apply ReLU activation to the output of the first layer
        x = torch.relu(self.fc2(x)) # Apply ReLU activation to the output of the second layer
        x = self.fc3(x)             # Pass through the output layer to get Q-values (no activation)
        return x                    # Return the Q-values for each action

# Define the ReplayMemory class for storing transitions
class ReplayMemory:             # Create a class to manage the replay memory
    def __init__(self, capacity):   # Initialize the memory with a specified capacity
        self.memory = deque(maxlen=capacity)  # Create a deque with fixed maximum length for transitions

    def push(self, transition):     # Method to add a transition to the memory
        self.memory.append(transition)  # Append the transition tuple (state, action, reward, next_state, done)

    def sample(self, batch_size):   # Method to sample a random batch of transitions
        return random.sample(self.memory, batch_size)  # Return a list of batch_size randomly sampled transitions

    def __len__(self):              # Method to get the current size of the memory
        return len(self.memory)     # Return the number of transitions currently stored

# Define function to select an action using epsilon-greedy policy
def select_action(state, epsilon, model, action_size): # Select action based on state, epsilon, model, and action space size
    if random.random() > epsilon:   # If random number exceeds epsilon, exploit (choose best action)
        with torch.no_grad():       # Disable gradient computation to save memory during inference
            q_values = model(state) # Compute Q-values for the current state using the model
            return q_values.argmax().item()  # Return the index of the action with the highest Q-value
    else:                           # Otherwise, explore (choose random action)
        return random.randrange(action_size)  # Return a random action index between 0 and action_size-1

# Define function to optimize the DQN model
def optimize_model(model, target_model, memory, optimizer): # Optimize the model using target network, memory, and optimizer
    if len(memory) < BATCH_SIZE:    # Check if memory has enough transitions for a batch
        return                      # Return early if not enough transitions are available
    transitions = memory.sample(BATCH_SIZE)  # Sample a random batch of BATCH_SIZE transitions

    # Extract components from the transitions
    batch_state = torch.cat([t[0] for t in transitions])  # Concatenate states into a tensor [batch_size, state_size]
    batch_action = torch.LongTensor([t[1] for t in transitions]).view(-1, 1)  # Create tensor of actions [batch_size, 1]
    batch_reward = torch.FloatTensor([t[2] for t in transitions])  # Create tensor of rewards [batch_size]
    batch_next_state = torch.cat([t[3] for t in transitions])  # Concatenate next states [batch_size, state_size]
    batch_done = torch.FloatTensor([t[4] for t in transitions])  # Create tensor of done flags [batch_size]

    # Compute current Q-values for the actions taken
    current_q_values = model(batch_state).gather(1, batch_action).squeeze(1)  # Get Q-values for selected actions [batch_size]

    # Compute target Q-values using the target network
    with torch.no_grad():           # Disable gradient computation for target Q-values
        next_q_values = target_model(batch_next_state).max(1)[0]  # Get max Q-values for next states [batch_size]
        target_q_values = batch_reward + (GAMMA * next_q_values * (1 - batch_done))  # Apply Bellman equation for targets

    # Compute the loss (mean squared error)
    loss = nn.MSELoss()(current_q_values, target_q_values)  # Calculate MSE between current and target Q-values

    # Perform optimization
    optimizer.zero_grad()           # Clear previous gradients in the optimizer
    loss.backward()                 # Compute gradients of the loss with respect to model parameters
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients to a max norm of 1.0 for stability
    optimizer.step()                # Update model weights using the optimizer

# Define function to visualize an episode
def visualize_episode(model, episode=0): # Visualize an episode using the model, with episode number for display
    env = gym.make("CartPole-v1", render_mode="human")  # Create a CartPole environment with human rendering
    print(f"Visualizing episode {episode+1}")  # Print the episode number being visualized
    state, _ = env.reset()  # Reset the environment and get the initial state
    state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension [1, state_size]
    done = False  # Initialize done flag to False
    total_reward = 0  # Initialize total reward for the visualized episode
    timeout = 3000  # Set maximum timesteps to prevent infinite loops
    timestep = 0  # Initialize timestep counter
    while (not done and timestep < timeout): # Continue until episode ends or timeout is reached
        with torch.no_grad():  # Disable gradient computation for visualization
            q_values = model(state)  # Compute Q-values for the current state
            action = q_values.argmax().item()  # Select action with the highest Q-value
        next_state, reward, done, _, _ = env.step(action)  # Perform action in the environment
        state = torch.FloatTensor(next_state).unsqueeze(0)  # Convert next state to tensor with batch dimension
        total_reward += reward  # Accumulate the reward
        timestep += 1  # Increment timestep counter
    print(f"Reward for visualized episode: {total_reward}")  # Print the total reward for the episode
    env.close()  # Close the environment and rendering window

# Define function to save the model and training state
def save_model(model, state_size, action_size, optimizer, episode, epsilon, rewards, filepath="model.pth"): # Save model and training state to a file
    """Save the model, optimizer, and training state to a file.
    
    Args:
        model: The DQN neural network.
        state_size: Dimension of the state space.
        action_size: Number of possible actions.
        optimizer: The Adam optimizer.
        episode: Current episode number.
        epsilon: Current epsilon value.
        rewards: List of accumulated episode rewards.
        filepath: Path to save the checkpoint.
    """
    model_data = {  # Create a dictionary to store model and training data
        'model_state_dict': model.state_dict(),  # Save the model's weights
        'state_size': state_size,  # Save the state space dimension
        'action_size': action_size,  # Save the action space size
        'optimizer_state_dict': optimizer.state_dict(),  # Save the optimizer's state
        'episode': episode,  # Save the current episode number
        'epsilon': epsilon,  # Save the current epsilon value
        'rewards': rewards  # Save the list of episode rewards
    }
    torch.save(model_data, filepath)  # Save the dictionary to the specified file
    print(f"Model saved to: {filepath}")  # Print confirmation of successful save

# Define function to load a saved model and training state
def load_model(filepath="model.pth"): # Load model and training state from a file
    """Load the model and training state from a file.
    
    Args:
        filepath: Path to the checkpoint file.
    
    Returns:
        model: Loaded DQN model.
        state_size: Dimension of the state space.
        action_size: Number of possible actions.
        optimizer: Loaded Adam optimizer.
        episode: Saved episode number.
        epsilon: Saved epsilon value.
        rewards: List of saved episode rewards.
    """
    if os.path.exists(filepath):  # Check if the checkpoint file exists
        model_data = torch.load(filepath, weights_only=False)  # Load the checkpoint dictionary
        state_size = model_data['state_size']  # Extract state space dimension
        action_size = model_data['action_size']  # Extract action space size
        model = DQN(state_size, action_size)  # Initialize a new DQN model
        model.load_state_dict(model_data['model_state_dict'])  # Load the saved model weights
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Initialize Adam optimizer
        optimizer.load_state_dict(model_data['optimizer_state_dict'])  # Load the saved optimizer state
        episode = model_data['episode']  # Extract the saved episode number
        epsilon = model_data['epsilon']  # Extract the saved epsilon value
        rewards = model_data['rewards']  # Extract the saved list of rewards
        print(f"Model loaded from: {filepath}")  # Print confirmation of successful load
        return model, state_size, action_size, optimizer, episode, epsilon, rewards  # Return loaded data
    else:  # If the file does not exist
        raise FileNotFoundError(f"No model found at: {filepath}")  # Raise an error

# Define the main training function
def train(model=None, optimizer=None, epsilon=None, start_episode=0, episode_rewards=[]): # Train the DQN agent with optional pre-loaded state
    # Initialize environment and networks
    env = gym.make("CartPole-v1")  # Create the CartPole-v1 environment
    state_size = env.observation_space.shape[0]  # Get state space dimension (4 for CartPole)
    action_size = env.action_space.n  # Get number of possible actions (2 for CartPole)

    if not model:  # Check if a model is provided
        model = DQN(state_size, action_size)  # Initialize a new DQN model if none provided
    target_model = DQN(state_size, action_size)  # Initialize the target DQN model
    target_model.load_state_dict(model.state_dict())  # Copy weights from main model to target model
    target_model.eval()  # Set target model to evaluation mode (no gradient updates)

    if not optimizer:  # Check if an optimizer is provided
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Initialize Adam optimizer if none provided
    memory = ReplayMemory(MEMORY_SIZE)  # Initialize replay memory with specified capacity

    # Initialize tracking variables
    if not epsilon:  # Check if epsilon is provided
        epsilon = EPSILON_START  # Set epsilon to initial value if none provided

    # Main training loop
    for episode in range(start_episode, EPISODES): # Iterate over episodes from start_episode to EPISODES
        state, _ = env.reset()  # Reset environment and get initial state
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor with batch dimension [1, state_size]
        total_reward = 0  # Initialize total reward for the episode
        done = False  # Initialize done flag to False

        while not done:  # Continue until the episode ends
            action = select_action(state, epsilon, model, action_size)  # Select an action using epsilon-greedy policy
            next_state, reward, done, _, _ = env.step(action)  # Perform action and get next state, reward, done
            next_state = torch.FloatTensor(next_state).unsqueeze(0)  # Convert next state to tensor with batch dimension
            total_reward += reward  # Accumulate the reward

            # Store transition in replay memory
            transition = (state, action, reward, next_state, done)  # Create transition tuple
            memory.push(transition)  # Add transition to replay memory
            state = next_state  # Update current state to next state
            optimize_model(model, target_model, memory, optimizer)  # Optimize the model using a batch of transitions

        # Update target network periodically
        if episode % TARGET_UPDATE == 0:  # Check if it's time to update the target network
            target_model.load_state_dict(model.state_dict())  # Copy weights from main model to target model

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)  # Reduce epsilon, but not below EPSILON_END

        # Save and print episode results
        episode_rewards.append(total_reward)  # Append episode reward to the list
        print(f"Episode {episode+1}/{EPISODES} | Reward: {total_reward} | Epsilon: {epsilon:.3f}")  # Print episode summary

        # Save checkpoint periodically
        if (episode + 1) % SAVE_CHECKPOINT_EVERY == 0:  # Check if it's time to save a checkpoint
            save_model(model, state_size, action_size, optimizer, episode, epsilon, episode_rewards, filepath="model.pth")  # Save model and state

        # Visualize episode periodically
        if (episode + 1) % VISUALIZE_EVERY == 0:  # Check if it's time to visualize an episode
            visualize_episode(model, episode)  # Visualize the agent's performance

    # Close the environment
    env.close()  # Close the environment to free resources

    # Visualize training results
    plt.plot(episode_rewards)  # Plot the rewards for each episode
    plt.title("Rewards per Episode")  # Set the plot title
    plt.xlabel("Episode")  # Set the x-axis label
    plt.ylabel("Total Reward")  # Set the y-axis label
    plt.grid(True)  # Add a grid for readability
    plt.show()  # Display the plot

    return model  # Return the trained model

# Main entry point
if __name__ == "__main__":  # Check if the script is run directly
    # Load a saved model and continue training
    model, state_size, action_size, optimizer, episode, epsilon, rewards = load_model()  # Load model and training state
    # Train the model, resuming from the loaded state
    model = train(model=model, optimizer=optimizer, epsilon=epsilon, start_episode=episode, episode_rewards=rewards)  # Resume training