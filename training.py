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

    def get_exp(self):
        """
        :return: returns a minibatch of the size of args.batch with random experiences
        """
        return random.sample(self.memory, self.args.batch)

class Training():
    """
    - Training class with inits
    - self.args: arguments from cmdl, necessary for training details.
    - self.replay_memory: the replay memory that is used to store and get experiences
    - self.env: the environment we have to perform reset() and step() to get information to calculate everything
    - self.q_network: network we are training with
    - self.t_network, target network
    - self.opti = optimer that we use
    - self.calc_loss: the type of loss we want to calculate
    - self.loss_curve: for showing the
    """
    def __init__(self, args):
        self.args = args
        self.replay_memory = ReplayMemory(args)
        self.env: gym.Env = args.env
        self.q_network = networks.QFunction(input_dim=7056, output_dim=len(self.env.action_space.n))
        self.t_network = networks.QFunction(input_dim=7056, output_dim=len(self.env.action_space.n))
        # Weights are random, not 0, weights at q and t different, have to set weights
        self.set_weights()
        self.opti = optim.Adam(self.q_network.parameters())
        self.calc_loss = nn.MSELoss()


    def set_weights(self):
        """sets the weights from the q_network to the target_network"""
        self.t_network.load_state_dict(self.q_network.state_dict())


    def start_training(self):
        """
        training methods written the same as in the pseudocode
        """
        exploration = 0
        # to count the number of explorations, after reaching the limit, we use eps.-greedy
        update_counter = 0
        # to count the number of steps we have to take before update q_network to t_network
        losses = []
        # to save the losses we calculated, after every use for loss_curve its empty
        self.envsteps = 0
        # to count the environment steps we did not in a episode, in a whole training process
        used_minibatches = 0
        # to count the used minibatches to aggregate over loss to calc loss_curve
        self.loss_curve = []
        # contains the mean of every 25th minibatch
        for episode in range(self.args.episodes):

            self.preprocessed = self.env.reset()
            # at every start (reset) the start sequence is equal to the next state sequence.
            # there is no pre preimage

            start_time = datetime.datetime.now()
            # episodes can run for x steps or y time

            time_for_episode_in_sec = 360
            # time episodes are allowed to take

            for t in range(1000):

                if datetime.datetime.now()-start_time >= datetime.timedelta(seconds=time_for_episode_in_sec): break
                # before going into a new step, we proof if time limit is reached for that episode.

                if (exploration < self.args.expl_frame) or (np.random.random() is self.args.final_expl):
                    exploration += 1
                    action = self.env.action_space.sample()
                # choose the action random if eps.-greedy or still in exploration phase

                else:
                    action = np.argmax(self.q_network(self.preprocessed))
                # if not eps.-greedy or exploration, we get take the best action that we can when we put it into
                # the q_network

                self.old_preprocessed = self.preprocessed
                # before get the new state (preprocessed image), we have to save the current
                # state as the old state.

                self.preprocessed, rew, terminated, info, done = self.env.step(action)
                self.envsteps += 1
                # perform action on environment to get the observation

                experience = [self.old_preprocessed, action, rew, self.preprocessed, done]
                self.replay_memory.save_exp(experience)
                # save the experience in replay_memory

                minibatches = self.replay_memory.get_exp()
                used_minibatches += 1
                # we take random experiences of size self.args.batch

                action_values, target_action_values = self.set_target_value(minibatches)
                # we set the target_action_value for every action_value

                loss = self.calculate_loss(action_values, target_action_values)
                losses.append(loss.detach().clone())
                # claculating the loss and append it to losses

                update_counter += 1
                # the first step is done, one closer to update
                if np.mod(update_counter, self.args.update_freq) == 0 :
                    self.set_weights()
                # done steps are the same as update_frequency, t_network has to be updated
                if terminated:
                    self.preprocessed = self.env.reset()
                # game is terminated, we have to reset it go keep going
                if used_minibatches%25 == 0:
                    self.loss_curve.append(torch.stack(losses).mean().item())
                    losses = []
                # 25 minibatches were used, we have to tage the mean of the curent losses

    def set_target_value(self, minibatches):
        """
        :param minibatches: the experiences to calculate the target values to
        :return: the old action value and the target values
        """
        rewards = torch.tensor([reward[0] for reward in minibatches], dtype=torch.float64)
        next_states = torch.tensor([nstate[0] for nstate in minibatches], dtype=torch.float64)
        dones = torch.tensor([done[0] for done in minibatches], dtype=torch.float64)

        with torch.no_grad():
            target_values = self.t_network(next_states).max(1)[0]
        # get the action_values for the next_states

        target_action_values = (self.args.discount * target_values * (1-dones)) + rewards
        # case 1: done is true, 1, no further rewards, just current rewards, y = x*(1-1)+r = r
        # case 2: not done, 0, further rewards, y = x + r

        return target_action_values

    def calculate_loss(self, target_action_value, minibatches):
        """ calculates the loss """
        states = torch.tensor([state[0] for state in minibatches], dtype=torch.float64)
        actions = torch.tensor([action[0] for action in minibatches], dtype=torch.float64)
        action_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squezze(1)
        # unsqueeze: dimension hinzugefügt für gather, erwartet tensor (x,1)
        # gather: Extrahieren der Q-Values
        # anschließende squeeze zum Erstellen der Alten Form
        # action value beinhaltet alle q_values für die aktion die in dem State ausgeführt wurde

        loss = self.calc_loss(action_values, target_action_value)
        self.opti.zero_grad()
        loss.backward()
        self.opti.step()
        return loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Random Agent in a Gymnasium environment.")
    parser.add_argument("--env", type=str, help="The environment to learn")
    parser.add_argument("--episodes", type=int, help="how many episodes")
    parser.add_argument("--batch", type=int, default=32, help="size of the minibatch. number of replays in it.")
    parser.add_argument("--replay", type=int, default=1000000, help="memory size, storage of replays")
    parser.add_argument("--update_freq", type=int, default=10000, help="frequenc to update q_network.")
    parser.add_argument("--discount", type=float, default=0.99, help="discount of future rewards")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--gradient_mom", type=int, default=10, help="The number of episodes to run.")
    # parser.add_argument("--init_expl", action="store_true", help="Log results to Weights & Biases.")
    # can be done in hard coded
    parser.add_argument("--final_expl", type=float, default=0.1, help="exploration rate after initial exploration phase.")
    parser.add_argument("--expl_frame", type=int, default=1000000, help="number of random action before using the best action")
    parser.add_argument("--replay_start_size", type=int, default=50000, help="starts using the replay memory when the storage has minimum x replays")
    parser.add_argument("--noop", type=int, default=30, help="number of noops that can be performed ")

    args = parser.parse_args()

    training = Training(args=args)
