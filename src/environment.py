import gymnasium as gym

def create_environment(render_mode=None):
    """Create the CartPole environment.

    Args:
        render_mode (str, optional): Rendering mode ('human' for visualization, None for training). Defaults to None.

    Returns:
        gym.Env: CartPole environment instance.
    """
    return gym.make("CartPole-v1", render_mode=render_mode)