from environment import DrugRecommendationEnv
from dqn_agent import DQNAgent
import numpy as np

def train_agent(num_episodes=1000, batch_size=32, gamma=0.95):
    """Train the DQN agent"""
    # Initialize environment and agent
    env = DrugRecommendationEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, 
                    epsilon=1.0,
                    epsilon_min=0.05,  # Increased minimum exploration
                    epsilon_decay=0.995)  # Slower decay
    
    # Training loop
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Get action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Train on batch if enough samples
            if len(agent.memory) > batch_size:
                agent.replay(batch_size, gamma)
        
        rewards.append(total_reward)
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward:.3f}, Epsilon: {agent.epsilon:.2f}")
    
    # Save the trained model
    agent.save('drug_recommendation_model.pth')
    print("\nModel saved to drug_recommendation_model.pth")
    
    # Evaluate the trained agent
    print("\nEvaluation Results:")
    eval_rewards = []
    for _ in range(100):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state, evaluate=True)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        eval_rewards.append(total_reward)
    
    print(f"Average Reward over 100 episodes: {np.mean(eval_rewards):.2f}")
    
    return agent

def evaluate_agent(agent, env, n_episodes=100):
    total_rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
        total_rewards.append(episode_reward)
    
    avg_reward = np.mean(total_rewards)
    print(f"\nEvaluation Results:")
    print(f"Average Reward over {n_episodes} episodes: {avg_reward:.2f}")
    
if __name__ == "__main__":
    # Train the agent
    trained_agent = train_agent()
    
    # Create new environment for evaluation
    eval_env = DrugRecommendationEnv()
    
    # Evaluate the trained agent
    evaluate_agent(trained_agent, eval_env)
