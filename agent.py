import networks
import torch

class DQNAgent():

    def __init__(self, **kwargs):
        print("Wello Horld")
        # initialisiere Netzwerk
        # dqn = Networks.DQN()
        # if kwargs is not Null ->

    def select_action(self, state):
        """Idea: use predict
        :param state: state to predict the action to
        :return: the action the agent should choose
        """
        print("Wello Horld")



    def compute_action_value(self, state, action):
        """Idea: predict the actions for the state, takes all outputs, search for
        the correct action and return it
        :param state: stated that is chosen
        :param action: action that is chosen
        :return: the action value for specific state and action
        """
        print("Wello Horld")

    def save_network(self, name):
        """ saves the network parameters as name
        :param name: of the save
        """
        print("Mini mini man, DQN")
        #torch.save(dqn.state_dict(), name)

    def load_network(self, name):
        """ load the network
        :param name: of the loaded network
        """
        print("Mini mini man, DQN")
        #torch.save(dqn.load_state_dict(torch.load(name)))



class PPOAgent():
    "Grundideen gleich wie bei DQNAgewnt"

    def __init__(self):
        print("wello Horld")

    def select_action(self, state):
        """Idea: use predict
        :param state: state to predict the action to
        :return: the action the agent should choose
        """
        print("xD")

    def compute_value(self, state):
        print("Wello Horld")

    def save_network(self, name):
        """ saves the network parameters as name
        :param name: of the save
        """
        print("Oh no no no, PPO")
        # torch.save(dqn.state_dict(), name)

    def load_network(self, name):
        """ load the network
        :param name: of the loaded network
        """
        print("Oh no no no, PPO")
        # torch.save(dqn.load_state_dict(torch.load(name)))

