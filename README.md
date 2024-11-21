# Drug Recommendation System using Reinforcement Learning

This project implements a drug recommendation system using Deep Q-Learning (DQN), a reinforcement learning algorithm. The system learns to recommend appropriate medications based on patient conditions and treatment outcomes.

## Project Structure

```
drug_recommendation/
├── environment.py      # Custom OpenAI Gym environment for drug recommendations
├── dqn_agent.py       # Implementation of the DQN agent
└── train.py           # Training and evaluation script
```

## Features

- Custom environment simulating patient-drug interactions
- Deep Q-Network (DQN) implementation with experience replay
- Patient state representation with 10 features (age, conditions, vitals, etc.)
- Synthetic drug effectiveness modeling
- Training and evaluation pipelines

## Requirements

- Python 3.7+
- PyTorch
- OpenAI Gym
- NumPy
- Other dependencies in requirements.txt

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To train the model:
```bash
python drug_recommendation/train.py
```

## Technical Details

### Environment
- State space: 10-dimensional continuous space representing patient features
- Action space: Discrete space of available drugs
- Reward: Based on drug effectiveness for patient condition

### DQN Agent
- Two-network architecture (main and target networks)
- Experience replay with fixed-size memory
- Epsilon-greedy exploration strategy
- Periodic target network updates

## Note

This is a simplified simulation for educational purposes. Real medical recommendations should always be made by qualified healthcare professionals.
