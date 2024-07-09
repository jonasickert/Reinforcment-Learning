import networks
import gymnasium as gym
import torch

class DQNAgent():
    
    #code von jonas 
    """
    def __init__(self, env:gym.Env, input_dim, **kwargs):
        print("Wello Horld")
        # initialisiere Netzwerk
        self.dqn = networks.QFunction(input_dim=input_dim, output_dim=len(env.action_space.shape))
        if kwargs:
            self.load_network(kwargs)
    """        
    #code von Maryem: removed len and .shape to test on CartPole-v1
    def __init__(self, env:gym.Env, input_dim, **kwargs):
        """
        Initializes the DQNAgent with the given environment and input dimension.

        Parameters:
        env (gym.Env): The environment in which the agent will operate.
        input_dim (int): The dimension of the input (observation space).

        """
        print("Wello Horld")
        # initialisiere Netzwerk
        self.dqn = networks.QFunction(input_dim=input_dim, output_dim=env.action_space.n)
        if kwargs:
            self.load_network(kwargs)

    #code von jonas 
    #def select_action(self, state):
        """Idea: use predict
        :param state: state to predict the action to
        :return: the action the agent should choose
        """
    #    return self.dqn(state)
        
    #code von Maryem:
    def select_action(self, state):
        """
        Selects an action based on the given state using the Deep Q-Network (DQN).

        Parameters:
        state (numpy.ndarray): The current state of the environment represented as a NumPy array.

        Returns:
        int: The selected action, represented as an integer.
        """
        state = torch.from_numpy(state).float().unsqueeze(0)  # Convert state to tensor

        # Disable gradient calculation since we are in evaluation mode
        with torch.no_grad():
            action_values = self.dqn(state)
        
        # Select the action with the highest value
        action = torch.argmax(action_values, dim=1).item()
        
        return action

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
        torch.save(self.dqn.state_dict(), name)

    def load_network(self, name):
        """ load the network
        :param name: of the loaded network
        """
        torch.load(self.dqn.load_state_dict(torch.load(name)))


