import argparse
import gymnasium as gym
import numpy as np
from pytorch_grad_cam import GradCAM

import videos
import wrapper
from random_agent import random_agent
#*****extend_auf4b*****
from agent import DQNAgent #Import the DQN agent 
import wandb
import torch
import ball_game

#enables the WB backend
wandb.require("core")

#*****extend_auf4b*****
#added agent_type
def evaluate_agent(env_name, agent_type, num_episodes, log_wandb, path):
    """
    Evaluate a random agent in a Gym environment over 10 episodes.

    Parameters:
    - env_name (str): Name of the Gym environment.
    - num_episodes (int): Number of episodes to run the evaluation (default is 10).
    - agent_type (str): random or dqn 
    - log_wandb (bool): if you want to log your results to wandb.ai
    
    """
    
    if log_wandb:
        wandb.init(project="Evaluations", entity="tudortmundg3-tu-dortmund")
        
    env = gym.make(env_name, render_mode="rgb_array")
    e = wrapper.Wrapper(envi=env)
    env = e.env
    # Initialize the random agent: from auf2b
    #agent = random_agent(env.action_space)

    #*****extend_auf4b*****
    if agent_type == 'random':
        agent = random_agent(env.action_space)
    elif agent_type == 'dqn':
        input_dim = env.observation_space.shape[0]
        agent = DQNAgent(env, input_dim, path, )
        agent.load_network(path)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")
    #**********************

    total_rewards = []

    for episode in range(num_episodes):
        # Reset the environment to start a new episode
        state, _ = env.reset()

        done = False
        episode_reward = 0
        steps = 0
        while not done:
            if steps > 1000:
               break
            steps += 1
            # Select an action randomly from class random_agent
            action = agent.select_action(state)
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        # Print the total reward for this episode
        print(f"Episode {episode + 1}: Total reward: {episode_reward}")
        
        #logs reward for average
        if log_wandb:
            print("waiting for logging")
            wandb.log({f"reward_episode_{episode + 1}": episode_reward})

    # Calculate the average reward over all episodes
    avg_reward = np.mean(total_rewards)
    # Print the average reward
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")
    
    #logs the avg reward
    if log_wandb:
        wandb.log({"average_reward": avg_reward})
        wandb.finish()

    if agent_type == 'dqn':
        output_dir = "./evaluation_videos"
        target_layers = agent.dqn.conv3
        cam = GradCAM(model=agent.dqn, target_layers=[target_layers])
        videos.create_video(env_name, agent, output_dir, cam, steps_done=avg_reward, num_episodes=1, render_mode='rgb_array',
                    agent_type='dqn', log_wandb=False)



def evaluate_trained_model(env_name, model_path, num_episodes=10):
    env: gym.Env = gym.make(env_name, render_mode="rgb_array")  # Maryem
    e = wrapper.Wrapper(envi=env)
    env = e.env
    input_dim = env.observation_space.shape[2]  # Maryem
    agent = DQNAgent(env, input_dim, None)
    print(model_path)
    agent.load_network(model_path)  # Load the trained model weights
    print("model loaded into agent")

    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        action_counter = 0
        old_action = None
        steps = 0
        while not done:
            if steps > 1000:
                break
            steps += 1
            action = agent.select_action(state)  # Select action using the trained DQN agent
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            state = next_state
        print("done")

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total reward: {episode_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {num_episodes} episodes: {avg_reward}")

    env.close()


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
    parser.add_argument("--path", type=str)

    # Parse the command-line arguments
    
    parser.add_argument("--log_wandb", action="store_true", help="Log results to Weights & Biases.")

    
    args = parser.parse_args()

    print(gym.registry.keys())

    #*****extend_auf4b*****
    #added agent_type
    evaluate_agent(args.env, args.agent, args.episodes, args.log_wandb, args.path)

    """
    Example usage in the terminal:
    - Evaluate in CartPole-v1 environment with default 10 episodes:
    python evaluate.py --env CartPole-v1

    or

    python evaluate.py --env CartPole-v1 --episodes 10
    python evaluate.py --env CartPole-v1 --agent dqn --episodes 10
    python evaluate.py --env CartPole-v1 --agent random --episodes 10 --log_wandb
    python evaluate.py --env CartPole-v1 --agent dqn --episodes 10 --log_wandb
    python evaluate.py --env CartPole-v1 --log_wand


    #*****extend_auf4b*****
    #the new commands 
    python evaluate.py --env CartPole-v1 --agent random  --episodes 10
    python evaluate.py --env CartPole-v1 --agent dqn  --episodes 10
    
    python evaluate.py --env CartPole-v1 --agent dqn --episodes 10 --log_wandb --path yourpath
    """