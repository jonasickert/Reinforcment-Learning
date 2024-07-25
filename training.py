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
import wrapper
from agent import DQNAgent #Maryem
from evaluate import evaluate_trained_model #Maryem
import Minesweeper

from videos import create_video
import wandb
import os

class ReplayMemory():
    """Replay Memory, storage of experiences"""
    def __init__(self, args):
        """initialize the memory by maximal size of the memory
        :param args: args from parser, maximal size, size = number of arrays in it
        """
        self.args = args
        self.memory = deque(maxlen=args.replay)

    def save_exp(self, exp):
        """
        Saves the experiences. if memory is full, popleft, oldest experience, and append new on right
        Has to be [state, action, reward, state, done], if not => no save of experience
        :param exp: experience that has to be saved
        """
        if exp.__len__() == 5:
            if len(self.memory) != self.args.replay:
                self.memory.append(exp)
            else:
                self.memory.popleft()
                self.memory.append(exp)
        else:
             print("too much or too low elements in experience")
        # Maryem:
        """
        if len(exp) == 5:
            self.memory.append(exp)
        else:
            print("Too many or too few elements in experience")"""

    def get_exp(self):
        """
        :return: returns a minibatch of the size of args.batch with random experiences
        """
        # jonas: return random.sample(self.memory, self.args.batch)
        # Maryem:
        if len(self.memory) < self.args.batch:
            return []  # Not enough samples yet
        return random.sample(self.memory, self.args.batch)

class Training:
    """
    - Training class with inits
    - self.args: arguments from cmdl, necessary for training details.
    - self.replay_memory: the replay memory that is used to store and get experiences
    - self.env: the environment we have to perform reset() and step() to get information to calculate everything
    - self.q_network: network we are training with
    - self.t_network, target network
    - self.opti = optimizer that we use
    - self.calc_loss: the type of loss we want to calculate
    - self.loss_curve: for showing the
    """
    def __init__(self, args):
        self.args = args
        self.replay_memory = ReplayMemory(args)
        # jonas: self.env: gym.Env = args.env
        if self.args.env == 'Minesweeper-features-v0.1':
            self.render_mode = ""
            obs_type = "features"
            self.env: gym.Env = gym.make(args.env, render_mode=self.render_mode)
            if obs_type == "features":
                agent_shape = self.env.observation_space["agent"].shape
                cells_shape = self.env.observation_space["cells"].shape
                self.input_dim = agent_shape[0] + cells_shape[0] * cells_shape[1]
        else:
            self.render_mode = "rgb_array"
            self.env: gym.Env = gym.make(args.env, render_mode=self.render_mode) #Maryem
            wrap = wrapper.Wrapper(envi=self.env)
            self.env = wrap.env
            self.input_dim = self.env.observation_space.shape[0] #Maryem
            if self.args.env == 'Minesweeper-pixels-v0.1':
                self.input_dim = self.env.observation_space.shape[2]  # Maryem

        self.output_dim = self.env.action_space.n #Maryem
        # jonas: self.q_network = networks.QFunction(input_dim=7056, output_dim=len(self.env.action_space.n))
        # jonas: self.t_network = networks.QFunction(input_dim=7056, output_dim=len(self.env.action_space.n))
        # Maryem:
        # for minesweeper pixel space
        self.q_network = networks.QFunction(input_dim=self.input_dim, output_dim=self.output_dim, input_type="pixels")
        self.t_network = networks.QFunction(input_dim=self.input_dim, output_dim=self.output_dim, input_type="pixels")
        if self.args.load is not None:
            e = torch.load(self.args.load)
            self.q_network.load_state_dict(e)
            print("load of model successful")
        # Weights are random, not 0, weights at q and t different, have to set weights
        self.set_weights()
        # jonas :self.opti = optim.Adam(self.q_network.parameters())
        self.opti = optim.Adam(self.q_network.parameters(), lr=self.args.learning_rate) #Maryem
        self.calc_loss = nn.MSELoss()
        self.batches_count = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.q_network = self.q_network.to(self.device)
        self.t_network = self.t_network.to(self.device)
        print(f'Using device: {self.device}')
        self.steps_done = 0
        wandb.init(project="dqn_training_minesweeper", config=args)
        self.filename = f"model_{self.render_mode}_{datetime.datetime.today().day}_{datetime.datetime.now().hour}{datetime.datetime.now().minute}.pth"



    def set_weights(self):
        self.t_network.load_state_dict(self.q_network.state_dict())


    def start_training(self):
        """
        training methods written the same as in the pseudocode
        """

        total_timesteps = 2500000
        evaluation_interval = 2000
        next_evaluation = evaluation_interval

        while self.steps_done < total_timesteps:
            state, _ = self.env.reset()

            total_reward = 0
            # at every start (reset) the start sequence is equal to the next state sequence.
            # there is no pre preimage

            start_time = datetime.datetime.now()
            episode_steps = 0
            # episodes can run for x steps or y time
            while episode_steps < 1000:  # Environment steps within one episode
                # if datetime.datetime.now() - start_time >= datetime.timedelta(seconds=360):
                #     break
                # before going into a new step, we proof if time limit is reached for that episode.

                self.steps_done += 1
                episode_steps += 1
                exploration_rate = self.args.final_expl + (1 - self.args.final_expl) * np.exp(-1. * self.steps_done / self.args.expl_frame)

                if np.random.random() < exploration_rate:
                    action = self.env.action_space.sample()
                # choose the action random if eps.-greedy or still in exploration phase

                else:
                    if self.args.env == 'Minesweeper-features-v0.1':
                        agent = state['agent']
                        cells = state['cells'].flatten()  # Flatten the cells array
                        combined = np.concatenate([agent, cells])
                        tensor = torch.tensor(combined, dtype=torch.float32)
                        state_tensor = tensor.unsqueeze(0)
                    else:
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                        #state_tensor = state_tensor.permute(0, 3, 1, 2)
                    with torch.no_grad():
                        action = np.argmax(self.q_network(state_tensor).cpu().numpy())

                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward

                experience = [state, action, reward, next_state, done]
                self.replay_memory.save_exp(experience)
                state = next_state

                minibatches = self.replay_memory.get_exp()
                if minibatches and self.args.expl_frame<=self.steps_done:
                    #
                    self.batches_count += 1
                    target_action_values = self.set_target_value(minibatches)
                    loss = self.calculate_loss(target_action_values, minibatches)
                    self.opti.zero_grad()
                    loss.backward()
                    self.opti.step()

                    if self.steps_done % self.args.update_freq == 0:
                        self.set_weights()

                    # Print the current episode, environment step, reward, and loss function
                    #print(f"Step: {self.steps_done}, Episode Step: {episode_steps}, Reward: {reward}, Loss: {loss.item()}")

                    if self.batches_count % 25 == 0:
                        wandb.log({"loss_every_25_batches": loss.item(), "steps_done": self.steps_done})

                if done:
                    break

            print(f"Total Reward: {total_reward}, Episode Steps: {episode_steps}, total steps: {self.steps_done}")
            wandb.log({"total_reward": total_reward, "steps_done": self.steps_done})

            # Check if it's time to evaluate the model
            print(self.steps_done>=next_evaluation)
            if self.steps_done >= next_evaluation:
                print("Evaluating the model...")
                if self.args.load is not None:
                    self.filename = self.args.load
                filename = f"model_{self.render_mode}_{datetime.datetime.today().day}_{datetime.datetime.now().hour}_{datetime.datetime.now().min}_{self.steps_done}.pth"
                torch.save(self.q_network.state_dict(), self.filename)
                print("saved")
                #mean_reward = evaluate_trained_model(self.args.env, filename, num_episodes=10)
                #wandb.log({"mean_reward": mean_reward, "steps_done": self.steps_done})
                #self.log_evaluation(self.args.env, filename, self.steps_done)
                next_evaluation += evaluation_interval

        print("Training completed.")

    def set_target_value(self, minibatches):
        """
        :param minibatches: the experiences to calculate the target values to
        :return: the old action value and the target values
        """
        # jonas: rewards = torch.tensor([reward[0] for reward in minibatches], dtype=torch.float64)
        # jonas: next_states = torch.tensor([nstate[0] for nstate in minibatches], dtype=torch.float64)
        # jonas: dones = torch.tensor([done[0] for done in minibatches], dtype=torch.float64)
        #jonas:
        if self.args.env == 'Minesweeper-features-v0.1':
            next_states = []
            for nstate in minibatches:
                agent = nstate[3]['agent']
                cells = nstate[3]['cells'].flatten()  # Flatten the cells array
                combined = np.concatenate([agent, cells])
                next_states.append(combined)
            next_states = torch.tensor(next_states, dtype=torch.float32)
        else:
            next_states = torch.tensor([nstate[3] for nstate in minibatches], dtype=torch.float32).to(self.device)
            #next_states = next_states.permute(0, 3, 1, 2)
        # Maryem:
        rewards = torch.tensor([reward[2] for reward in minibatches], dtype=torch.float32).to(self.device)
        dones = torch.tensor([done[4] for done in minibatches], dtype=torch.float32).to(self.device)

        if next_states.shape[0] == 0: #Maryem
            print("No next states available. Skipping target value computation.") #Maryem
            return torch.tensor([]) #Maryem

        with torch.no_grad():
            # jonas: target_values = self.t_network(next_states).max(1)[0]
            # Maryem:
            #next_states = next_states.view(len(next_states), -1)  # Reshape to (batch_size, input_dim)
            target_values = self.t_network(next_states).max(1)[0]
        # get the action_values for the next_states

        target_action_values = (self.args.discount * target_values * (1 - dones)) + rewards
        # case 1: done is true, 1, no further rewards, just current rewards, y = x*(1-1)+r = r
        # case 2: not done, 0, further rewards, y = x + r

        return target_action_values

    def calculate_loss(self, target_action_values, minibatches):
        # jonas: states = torch.tensor([state[0] for state in minibatches], dtype=torch.float64)
        # jonas: actions = torch.tensor([action[0] for action in minibatches], dtype=torch.float64)
        # jonas: action_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squezze(1)
        # unsqueeze: dimension hinzugefügt für gather, erwartet tensor (x,1)
        # gather: Extrahieren der Q-Values
        # anschließende squeeze zum Erstellen der Alten Form
        # action value beinhaltet alle q_values für die aktion die in dem State ausgeführt wurde
        # jonas: loss = self.calc_loss(action_values, target_action_value)
        # Maryem:
        if len(target_action_values) == 0:
            print("No target action values available. Skipping loss computation.")
            return torch.tensor(0.0)
        if self.args.env == 'Minesweeper-features-v0.1':
            states = []
            for state in minibatches:
                agent = state[0]['agent']
                cells = state[0]['cells'].flatten()  # Flatten the cells array
                combined = np.concatenate([agent, cells])
                states.append(combined)
            states = torch.tensor(states, dtype=torch.float32)
        else:
            states = torch.tensor([state[0] for state in minibatches], dtype=torch.float32).to(self.device)
        actions = torch.tensor([action[1] for action in minibatches], dtype=torch.int64).to(self.device)
        #states = states.view(len(states), -1)  # Reshape to (batch_size, input_dim)
        target_action_values = target_action_values.to(self.device)
        action_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.calc_loss(action_values, target_action_values)
        wandb.log({"loss": loss.item(), "steps_done": self.steps_done})
        return loss

    def log_evaluation(self, env_name, model_path, steps_done):
        output_dir = "./videos"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        agent = DQNAgent(self.env, self.input_dim, None)
        agent.load_network(model_path)
        create_video(env_name, agent, output_dir,steps_done=steps_done,num_episodes=1, render_mode='rgb_array', agent_type='dqn', log_wandb=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent.")
    parser.add_argument("--env", type=str, help="The environment to learn")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--batch", type=int, default=32, help="Minibatch size")
    parser.add_argument("--replay", type=int, default=1000000, help="Replay memory size")
    parser.add_argument("--update_freq", type=int, default=5000, help="Frequency to update the target network")
    parser.add_argument("--discount", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--expl_frame", type=int, default=100000, help="Number of frames to perform exploration")
    parser.add_argument("--final_expl", type=float, default=0.1, help="Final exploration rate")
    parser.add_argument("--replay_start_size", type=int, default=50000, help="Minimum number of replay memories before training starts")
    parser.add_argument("--noop", type=int, default=30, help="Number of no-ops to perform")
    parser.add_argument("--load", type=str, default=None, help="put in path name")

    args = parser.parse_args()

    training = Training(args=args)
    #Maryem 
    training.start_training()
#example use

#python training.py --env CartPole-v1 --episodes 10 --learning_rate 0.0005 --update_freq 5000 --expl_frame 50000 --final_expl 0.05
