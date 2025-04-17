import argparse
import yaml
from src.train import train
from src.utils import load_model, visualize_episode

def main():
    """Main entry point for the CartPole DRL project."""
    parser = argparse.ArgumentParser(description="CartPole Deep Reinforcement Learning Project")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint")
    parser.add_argument("--visualize", action="store_true", help="Visualize a single episode")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.train:
        if args.resume:
            # Load model and training state
            model, state_size, action_size, optimizer, episode, epsilon, rewards = load_model()
            # Resume training from the loaded episode
            train(config, model=model, optimizer=optimizer, epsilon=epsilon, start_episode=episode + 1, episode_rewards=rewards)
        else:
            # Start training from scratch
            train(config)
    elif args.visualize:
        # Load model for visualization
        model, _, _, _, _, _, _ = load_model()
        visualize_episode(model, episode=0)

if __name__ == "__main__":
    main()