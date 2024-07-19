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
from agent import DQNAgent #Maryem
from evaluate import evaluate_trained_model #Maryem


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
        # jonas: if exp.__len__() == 5:
        # jonas: if len(self.memory) != self.args.replay:
        # jonas:     self.memory.append(exp)
        # jonas: else:
        # jonas:     self.memory.popleft()
        # jonas:     self.memory.append(exp)
        # jonas: else:
        # jonas:     print("too much or too low elements in experience")
        # Maryem:
        if len(exp) == 5:
            self.memory.append(exp)
        else:
            print("Too many or too few elements in experience")

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
        self.env: gym.Env = gym.make(args.env) #Maryem
        self.input_dim = self.env.observation_space.shape[0] #Maryem
        self.output_dim = self.env.action_space.n #Maryem
        # jonas: self.q_network = networks.QFunction(input_dim=7056, output_dim=len(self.env.action_space.n))
        # jonas: self.t_network = networks.QFunction(input_dim=7056, output_dim=len(self.env.action_space.n))
        # Maryem:
        self.q_network = networks.QFunction(input_dim=self.input_dim, output_dim=self.output_dim)
        self.t_network = networks.QFunction(input_dim=self.input_dim, output_dim=self.output_dim)
        # Weights are random, not 0, weights at q and t different, have to set weights
        self.set_weights()
        # jonas :self.opti = optim.Adam(self.q_network.parameters())
        self.opti = optim.Adam(self.q_network.parameters(), lr=self.args.learning_rate) #Maryem
        self.calc_loss = nn.MSELoss()
        self.steps_done = 0

    def set_weights(self):
        self.t_network.load_state_dict(self.q_network.state_dict())

class Training:
    def __init__(self, args):
        self.args = args
        self.replay_memory = ReplayMemory(args)
        self.env = gym.make(args.env)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n
        self.q_network = networks.QFunction(input_dim=self.input_dim, output_dim=self.output_dim)
        self.t_network = networks.QFunction(input_dim=self.input_dim, output_dim=self.output_dim)
        self.set_weights()
        self.opti = optim.Adam(self.q_network.parameters(), lr=self.args.learning_rate)
        self.calc_loss = nn.MSELoss()
        self.steps_done = 0

    def set_weights(self):
        self.t_network.load_state_dict(self.q_network.state_dict())

    def start_training(self):
        for episode in range(self.args.episodes):
            state, _ = self.env.reset()
            total_reward = 0
            
            start_time = datetime.datetime.now()

            for t in range(1000):
                if datetime.datetime.now() - start_time >= datetime.timedelta(seconds=360):
                    break

                self.steps_done += 1
                exploration_rate = self.args.final_expl + (1 - self.args.final_expl) * np.exp(-1. * self.steps_done / self.args.expl_frame)

                if np.random.random() < exploration_rate:
                    action = self.env.action_space.sample()
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        action = np.argmax(self.q_network(state_tensor).numpy())

                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward

                experience = [state, action, reward, next_state, done]
                self.replay_memory.save_exp(experience)
                state = next_state

                minibatches = self.replay_memory.get_exp()
                if minibatches:
                    target_action_values = self.set_target_value(minibatches)
                    loss = self.calculate_loss(target_action_values, minibatches)
                    self.opti.zero_grad()
                    loss.backward()
                    self.opti.step()

                    if self.steps_done % self.args.update_freq == 0:
                        self.set_weights()
                    print(f"Episode: {episode}, Step: {t}, Loss: {loss.item()}")    

                    if done:
                        break

            print(f"Episode: {episode}, Total Reward: {total_reward}")

            if (episode + 1) % 100 == 0:  # Evaluate every 10 episodes
                print("Evaluating the model...")
                filename = f"final_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                torch.save(self.q_network.state_dict(), filename)
                evaluate_trained_model(self.args.env, filename, num_episodes=10) 
        
        print("Training completed.")

    def set_target_value(self, minibatches):
        """
        :param minibatches: the experiences to calculate the target values to
        :return: the old action value and the target values
        """
        # jonas: rewards = torch.tensor([reward[0] for reward in minibatches], dtype=torch.float64)
        # jonas: next_states = torch.tensor([nstate[0] for nstate in minibatches], dtype=torch.float64)
        # jonas: dones = torch.tensor([done[0] for done in minibatches], dtype=torch.float64)
        # Maryem:
        rewards = torch.tensor([reward[2] for reward in minibatches], dtype=torch.float32)
        next_states = torch.tensor([nstate[3] for nstate in minibatches], dtype=torch.float32)
        dones = torch.tensor([done[4] for done in minibatches], dtype=torch.float32)

        if next_states.shape[0] == 0: #Maryem
            print("No next states available. Skipping target value computation.") #Maryem
            return torch.tensor([]) #Maryem

        with torch.no_grad():
            # jonas: target_values = self.t_network(next_states).max(1)[0]
            # Maryem:
            next_states = next_states.view(len(next_states), -1)  # Reshape to (batch_size, input_dim)
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

        states = torch.tensor([state[0] for state in minibatches], dtype=torch.float32)
        actions = torch.tensor([action[1] for action in minibatches], dtype=torch.int64)
        states = states.view(len(states), -1)  # Reshape to (batch_size, input_dim)
        action_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.calc_loss(action_values, target_action_values)
        return loss



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
#example use 

#python training.py --env CartPole-v0 --episodes 5000 --learning_rate 0.0005 --update_freq 5000 --expl_frame 50000 --final_expl 0.05