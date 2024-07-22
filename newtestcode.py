import argparse
import datetime
import random
from collections import deque
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import networks
from agent import DQNAgent
from evaluate import evaluate_trained_model
from videos import create_video
import wandb
import os

class ReplayMemory():
    def __init__(self, args):
        self.args = args
        self.memory = deque(maxlen=args.replay)

    def save_exp(self, exp):
        if len(exp) == 5:
            self.memory.append(exp)
        else:
            print("Invalid experience format")

    def get_exp(self):
        if len(self.memory) < self.args.batch:
            return []  # Not enough samples yet
        return random.sample(self.memory, self.args.batch)

class Training:
    def __init__(self, args):
        self.args = args
        self.replay_memory = ReplayMemory(args)

        self.env: gym.Env = gym.make(args.env)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n

        self.q_network = networks.QFunction(input_dim=self.input_dim, output_dim=self.output_dim)
        self.t_network = networks.QFunction(input_dim=self.input_dim, output_dim=self.output_dim)

        self.opti = optim.Adam(self.q_network.parameters(), lr=self.args.learning_rate)
        self.calc_loss = nn.MSELoss()
        self.batches_count = 0
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f'Using device: {self.device}')
        self.steps_done = 0

        wandb.init(project="dqn_training", config=args)

    def set_weights(self):
        self.t_network.load_state_dict(self.q_network.state_dict())

    def start_training(self):
        evaluation_steps = 20  # Evaluate every 25,000 steps
        next_evaluation = evaluation_steps
        episode_number = 0
        
        while episode_number < self.args.episodes:  # Run for a specified number of episodes
            state, _ = self.env.reset()
            total_reward = 0
            start_time = datetime.datetime.now()
            episode_steps = 0
            episode_number += 1

            while episode_steps < 1000:  # Environment steps within one episode
                if datetime.datetime.now() - start_time >= datetime.timedelta(seconds=360):
                    break

                self.steps_done += 1
                episode_steps += 1
                exploration_rate = self.args.final_expl + (1 - self.args.final_expl) * np.exp(-1. * self.steps_done / self.args.expl_frame)

                if np.random.random() < exploration_rate:
                    action = self.env.action_space.sample()
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        action = np.argmax(self.q_network(state_tensor).cpu().numpy())

                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward

                experience = [state, action, reward, next_state, done]
                self.replay_memory.save_exp(experience)
                state = next_state

                minibatches = self.replay_memory.get_exp()
                if minibatches:
                    self.batches_count += 1
                    target_action_values = self.set_target_value(minibatches)
                    loss = self.calculate_loss(target_action_values, minibatches)
                    self.opti.zero_grad()
                    loss.backward()
                    self.opti.step()

                    if self.steps_done % self.args.update_freq == 0:
                        self.set_weights()
                    
                    # Print the current episode, environment step, reward, and loss function
                    print(f"Episode: {episode_number}, Step: {episode_steps}, Reward: {reward}, Loss: {loss.item()}")

                    if self.batches_count % 25 == 0:
                        wandb.log({"loss_every_25_batches": loss.item(), "steps_done": self.steps_done})

                if done:
                    break

            print(f"Episode: {episode_number}, Total Reward: {total_reward}, Episode Steps: {episode_steps}")
            wandb.log({"total_reward": total_reward, "steps_done": self.steps_done})

            # Check if it's time to evaluate the model
            if self.steps_done >= next_evaluation:
                print("Evaluating the model...")
                filename = f"model_{self.steps_done}.pth"
                torch.save(self.q_network.state_dict(), filename)
                mean_reward = evaluate_trained_model(self.args.env, filename, num_episodes=10)
                wandb.log({"mean_reward": mean_reward, "steps_done": self.steps_done})
                self.log_evaluation(self.args.env, filename)
                next_evaluation += evaluation_steps

        print("Training completed.")


    def set_target_value(self, minibatches):
        rewards = torch.tensor([reward[2] for reward in minibatches], dtype=torch.float32)
        next_states = torch.tensor([nstate[3] for nstate in minibatches], dtype=torch.float32)
        dones = torch.tensor([done[4] for done in minibatches], dtype=torch.float32)

        if next_states.shape[0] == 0:
            print("No next states available. Skipping target value computation.")
            return torch.tensor([])

        with torch.no_grad():
            next_states = next_states.view(len(next_states), -1).to(self.device)
            target_values = self.t_network(next_states).max(1)[0]

        target_action_values = (self.args.discount * target_values * (1 - dones)) + rewards.to(self.device)
        return target_action_values

    def calculate_loss(self, target_action_values, minibatches):
        if len(target_action_values) == 0:
            print("No target action values available. Skipping loss computation.")
            return torch.tensor(0.0)

        states = torch.tensor([state[0] for state in minibatches], dtype=torch.float32).to(self.device)
        actions = torch.tensor([action[1] for action in minibatches], dtype=torch.int64).to(self.device)
        states = states.view(len(states), -1)
        action_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.calc_loss(action_values, target_action_values)
        wandb.log({"loss": loss.item(), "steps_done": self.steps_done})
        return loss

    def log_evaluation(self, env_name, model_path):
        output_dir = "./videos"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        agent = DQNAgent(self.env, self.input_dim)
        agent.load_network(model_path)
        create_video(env_name, agent, output_dir, num_episodes=1, render_mode='rgb_array', agent_type='dqn', log_wandb=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent.")
    parser.add_argument("--env", type=str, help="The environment to learn")
    parser.add_argument("--episodes", type=int, help="Number of episodes")
    parser.add_argument("--batch", type=int, default=32, help="Minibatch size")
    parser.add_argument("--replay", type=int, default=1000000, help="Replay memory size")
    parser.add_argument("--update_freq", type=int, default=10000, help="Frequency to update the target network")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--expl_frame", type=int, default=1000000, help="Number of frames to perform exploration")
    parser.add_argument("--final_expl", type=float, default=0.1, help="Final exploration rate")
    parser.add_argument("--replay_start_size", type=int, default=50000, help="Minimum number of replay memories before training starts")
    parser.add_argument("--noop", type=int, default=30, help="Number of no-ops to perform")

    args = parser.parse_args()

    training = Training(args=args)
    training.start_training()

#python newtestcode.py --env CartPole-v1 --episodes 10 --learning_rate 0.0005 --update_freq 5000 --expl_frame 50000 --final_expl 0.05
