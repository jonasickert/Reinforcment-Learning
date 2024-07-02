import argparse
import gymnasium as gym
import numpy as np
from random_agent import random_agent


def evaluate_agent(env_name, num_episodes=10):
    env = gym.make(env_name)
    agent = random_agent(env.action_space)
    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total reward: {episode_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")


# Example to write in the terminal :python evaluate.py --env CartPole-v1
# or
# python evaluate.py --env CartPole-v1 --episodes 10
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Random Agent in a Gymnasium environment.")
    parser.add_argument("--env", type=str, required=True, help="The Gymnasium environment to use.")
    parser.add_argument("--episodes", type=int, default=10, help="The number of episodes to run.")

    args = parser.parse_args()
    evaluate_agent(args.env, args.episodes)
