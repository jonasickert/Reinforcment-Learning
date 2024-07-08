import argparse
import gymnasium as gym
import numpy as np
from random_agent import random_agent
#*****extend_auf4b*****
from agent import DQNAgent #Import the DQN agent 

#*****extend_auf4b*****
#added agent_type
def evaluate_agent(env_name, agent_type='random', num_episodes=10):
    """
    Evaluate a random agent in a Gym environment over 10 episodes.

    Parameters:
    - env_name (str): Name of the Gym environment.
    - num_episodes (int): Number of episodes to run the evaluation (default is 10).
    """
    env = gym.make(env_name)
    # Initialize the random agent: from auf2b
    #agent = random_agent(env.action_space)

    #*****extend_auf4b*****
    if agent_type == 'random':
        agent = random_agent(env.action_space)
    elif agent_type == 'dqn':
        input_dim = env.observation_space.shape[0]
        agent = DQNAgent(env, input_dim)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    #**********************

    total_rewards = []

    for episode in range(num_episodes):
        # Reset the environment to start a new episode
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Select an action randomly from class random_agent
            action = agent.select_action(state)
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        # Print the total reward for this episode
        print(f"Episode {episode + 1}: Total reward: {episode_reward}")

    # Calculate the average reward over all episodes
    avg_reward = np.mean(total_rewards)
    # Print the average reward
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")



if __name__ == "__main__":
    # Create an argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a Random Agent in a Gymnasium environment.")
    # Define the argument for the Gym environment name
    parser.add_argument("--env", type=str, required=True, help="The Gymnasium environment to use.")

    #*****extend_auf4b*****
    parser.add_argument("--agent", type=str, choices=["random", "dqn"], default="random", help="The type of agent to use.")
    #**********************

    # Define the argument for the number of episodes
    parser.add_argument("--episodes", type=int, default=10, help="The number of episodes to run.")
    # Parse the command-line arguments
    args = parser.parse_args()

    #*****extend_auf4b*****
    #added agent_type
    evaluate_agent(args.env, args.agent, args.episodes)

    """
    Example usage in the terminal:
    - Evaluate in CartPole-v1 environment with default 10 episodes:
    python evaluate.py --env CartPole-v1

    or

    python evaluate.py --env CartPole-v1 --episodes 10
    python evaluate.py --env CartPole-v1 --agent dqn --episodes 10

    #*****extend_auf4b*****
    #the new commands 
    python evaluate.py --env CartPole-v1 --agent random  --episodes 10
    python evaluate.py --env CartPole-v1 --agent dqn  --episodes 10
    """