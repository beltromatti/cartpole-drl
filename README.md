# CartPole DRL Project

A Deep Reinforcement Learning (DRL) project implementing a Deep Q-Network (DQN) to solve the CartPole-v1 environment from Gymnasium.

## Overview

This project trains a DQN agent to balance a pole on a cart using the CartPole-v1 environment. The agent learns through experience replay and an epsilon-greedy policy, with periodic visualization and model checkpointing.

## Author

This project was developed by Mattia Beltrami, a student of Computer Science for Management at the University of Bologna (UNIBO).

## Project Structure

```
cartpole-drl/
├── src/
│   ├── __init__.py           # Empty file to make src a Python module
│   ├── model.py              # DQN model definition
│   ├── memory.py             # Replay memory implementation
│   ├── utils.py              # Utility functions (action selection, optimization, visualization, saving/loading)
│   ├── train.py              # Training logic
│   └── environment.py        # Environment creation
├── config/
│   └── config.yaml           # Configuration parameters
├── data/
│   ├── checkpoints/          # Model checkpoints
│   └── results/              # Training plots
├── run.py                    # Main script
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
├── LICENSE                   # License file
└── .gitignore                # Git ignore file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cartpole-drl.git
   cd cartpole-drl
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
Train a new model from scratch:
```bash
python run.py --train
```

### Resuming Training
Resume training from a saved checkpoint:
```bash
python run.py --train --resume
```

### Visualization
Visualize a single episode using a saved model:
```bash
python run.py --visualize
```

### Configuration
Edit `config/config.yaml` to adjust parameters such as the number of episodes, learning rate, or visualization frequency.

## Notes

* The default configuration in `config/config.yaml` is designed to train the agent effectively, typically achieving stable performance after 350-450 episodes.  
* Training duration may vary depending on hardware and random seed. For consistent results, consider averaging rewards over multiple runs.  
* To improve training stability, ensure the replay memory is sufficiently large (default: 20,000 transitions) and adjust the learning rate if convergence is too slow or unstable.  
* Visualization can slow down training. Set `visualize_every` to a higher value (e.g., 100) for faster training on lower-end hardware.  
* Checkpoints are saved every 50 episodes by default. Use the `--resume` flag to continue training from the latest checkpoint.

## Use of Generative AI

This project leveraged generative artificial intelligence, specifically Grok 3 developed by xAI, to assist in generating comments and documentation. The AI was used to create clear, detailed, and educational explanations for the code and configuration files, enhancing the project's clarity for learning purposes. However, every line of code, comment, and documentation was carefully reviewed and validated by the author to ensure accuracy and correctness.

## Requirements

* Python 3.8+
* PyTorch 2.0.0+
* Gymnasium 0.29.1+
* NumPy
* Matplotlib
* PyYAML

See `requirements.txt` for full details.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.