import torch
import matplotlib.pyplot as plt
from .environment import create_environment
from .model import DQN
from .memory import ReplayMemory
from .utils import select_action, optimize_model, visualize_episode, save_model

def train(config, model=None, optimizer=None, epsilon=None, start_episode=0, episode_rewards=[]):
    """Train a DQN agent on the CartPole environment.

    Args:
        config (dict): Configuration parameters (episodes, batch_size, etc.).
        model (nn.Module, optional): Pre-initialized DQN model. Defaults to None.
        optimizer (optim.Optimizer, optional): Pre-initialized optimizer. Defaults to None.
        epsilon (float, optional): Initial epsilon value. Defaults to None.
        start_episode (int, optional): Starting episode number (for resuming). Defaults to 0.

    Returns:
        nn.Module: Trained DQN model.
    """
    # Initialize environment
    env = create_environment()
    state_size = env.observation_space.shape[0]  # State space dimension (4 for CartPole)
    action_size = env.action_space.n            # Number of actions (2 for CartPole)

    # Initialize model if not provided
    if model is None:
        model = DQN(state_size, action_size)
    target_model = DQN(state_size, action_size)  # Initialize target model
    target_model.load_state_dict(model.state_dict())  # Copy weights from main model
    target_model.eval()  # Set target model to evaluation mode

    # Initialize optimizer if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Initialize replay memory
    memory = ReplayMemory(config['memory_size'])

    # Initialize epsilon if not provided
    if epsilon is None:
        epsilon = config['epsilon_start']

    # Training loop
    for episode in range(start_episode, config['episodes']):
        state, _ = env.reset()  # Reset environment
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor [1, state_size]
        total_reward = 0
        done = False

        while not done:
            # Select action using epsilon-greedy policy
            action = select_action(state, epsilon, model, action_size)
            # Perform action in environment
            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)  # Convert next state to tensor
            total_reward += reward

            # Store transition in replay memory
            transition = (state, action, reward, next_state, done)
            memory.push(transition)
            state = next_state  # Update current state

            # Optimize model using a batch of transitions
            optimize_model(model, target_model, memory, optimizer, config['batch_size'], config['gamma'])

        # Update target model periodically
        if episode % config['target_update'] == 0:
            target_model.load_state_dict(model.state_dict())

        # Decay epsilon
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])

        # Store and print episode results
        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{config['episodes']} | Reward: {total_reward} | Epsilon: {epsilon:.3f}")

        # Save checkpoint periodically
        if (episode + 1) % config['save_checkpoint_every'] == 0:
            save_model(model, state_size, action_size, optimizer, episode, epsilon, episode_rewards)
            
        # Visualize episode periodically
        if (episode + 1) % config['visualize_every'] == 0:
            visualize_episode(model, episode)

    # Close environment
    env.close()

    # Plot and display training results
    plt.plot(episode_rewards)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig("data/results/rewards_plot.png")  # Save plot
    plt.show()

    return model