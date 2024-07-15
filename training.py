import argparse
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
        """
        initialize the memory by maximal size of the storage
        :param args: args from parser
        """
        self.args = args
        self.memory = deque(maxlen=args.replay)

    def save_exp(self, exp):
        """
        saves the experiences. if memory is full, popleft, oldest experience, and append new on right
        :param exp: experience to save
        """
        if len(self.memory) != self.args.replay:
            self.memory.append(exp)
        else:
            self.memory.popleft()
            #self.summarizeReplays()
            self.memory.append(exp)

    def summarizeReplays(self):
        """
        Idea: how to summarize? Replays are kind of sorted in the replay memory, summarizing neighbors
        can create summarizers of replays which are kind of same.
        How many should be summarized at once? 2, 10, 50 at a time? Has to be efficient!
        Newer experiences are "better" the network has learned, the rewards are better, old experiences
        are not as good.
        Deque works as a stack. Possible to push left.
        - Take 10 last experiences summarize them
        - pushleft => experience is at the bottom of the stack
        - When summarizing the next 10 experiences, the new experience contains the last experience
        - and so on
        """
        old_exp = [self.memory.popleft() for i in range(10)]
        agg_state = np.mean([exp[0] for exp in old_exp], axis=0)
        # state is returned pixel array, npmean aggregate over every single pixel from every expereince,
        # for example, every (x,y) pixel for experience 1 to 10. agg_state is a pixel array
        agg_action = old_exp[-1][1]  # Letzte Aktion beibehalten oder eine Kombination verwenden
        # takles action from last experience
        agg_reward = np.sum([exp[2] for exp in old_exp])
        agg_next_state = np.mean([exp[3] for exp in old_exp], axis=0)

        agg_exp = (agg_state, agg_action, agg_reward, agg_next_state)
        self.memory.appendleft(agg_exp)

    def get_exp(self):
        return random.sample(self.memory, self.args.batch)

class Training():
    """
    Training class with inits
    """
    def __init__(self, args):
        self.args = args
        self.replay_memory = ReplayMemory(args)
        self.env: gym.Env = args.env
        self.q_network = networks.QFunction(input_dim=7056, output_dim=len(self.env.action_space.n))
        self.t_network = networks.QFunction(input_dim=7056, output_dim=len(self.env.action_space.n))
        # Weights are random, not 0, weights at q and t different, havew to set weights
        self.set_weights()
        self.opti = optim.Adam(self.q_network.parameters())
        self.calc_loss = nn.MSELoss()

    def set_weights(self):
        """
        sets the weights from the q_network to the target_network
        """
        self.t_network.load_state_dict(self.q_network.state_dict())


    def training(self):
        """
        training methods written the same as in the pseudocode
        """
        exploration = 0
        # to count the number of explorations
        action = self.env.action_space.sample()
        # declare of the action
        for episode in range(self.args.episodes):
            self.preprocessed = self.env.reset()
            # at every start (reset) the start sequence is equal to the next state sequence.
            for t in range(1000):
                if (exploration < self.args.expl_frame) or (np.random.random() is self.args.final_expl):
                    # exploration phase not over or epsilon policy
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_network(self.preprocessed))
                self.old_preprocessed = self.preprocessed
                # before get the new state (preprocessed image), we have to save the current
                # state as the old state.
                self.preprocessed, rew, terminated, info, done = self.env.step(action)
                experience = [self.old_preprocessed, action, rew, self.preprocessed]
                self.replay_memory.save_exp(experience)
                minibatches = self.replay_memory.get_exp()
                action_values, target_values = self.set_target_value(minibatches)
                self.calculate_loss(action_values, target_values)
                #
                if np.mod(t, self.args.update_freq) == 0 :
                    self.set_weights()

    def set_target_value(self, minibatches):
        """
        :param minibatches: the experiences to calculate the target values to
        :return: the old action value and the target values
        """
        states = torch.cat(minibatches[0])
        states = torch.FloatTensor(states)
        actions = torch.cat(minibatches[1])
        actions = torch.LongTensor(actions)
        rewards = torch.cat(minibatches[2])
        rewards = torch.FloatTensor(rewards)
        next_states = torch.cat(minibatches[3])
        next_states = torch.FloatTensor(next_states)

        action_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squezze(1)
        # unsquezze: dimension hinzugefügt für gather, erwartet tensor (x,1)
        # gather: Extrahieren der Q-Values
        # anschließende squezze zum erstellen der Alten Form
        # action value beinhaltet alle q_values für die aktion die in dem State ausgeführt wurde

        target_values = self.t_network(next_states).max(1)[0]
        return action_values, target_values

    def calculate_loss(self, action_values, target_value):
        """
        calculates the loss
        """
        loss = self.calc_loss(action_values, target_value)
        self.opti.zero_grad()
        loss.backward()
        self.opti.step()



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
