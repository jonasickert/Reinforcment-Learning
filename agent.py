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
    
    #code von Jonas
    #def compute_action_value(self, state, action):
        """Idea: predict the actions for the state, takes all outputs, search for
        the correct action and return it
        :param state: stated that is chosen
        :param action: action that is chosen
        :return: the action value for specific state and action
        """
    #    print("Wello Horld")

    #code von Maryem for task 4)b
    def compute_action_value(self, state, action):
        """
        Computes the Q-value for a given state-action pair using the Deep Q-Network (DQN).

        Parameters:
        state (numpy.ndarray): The current state of the environment represented as a NumPy array.
        action (int): The action for which to compute the Q-value.

        Returns:
        float: The Q-value for the given state-action pair.
        """

        # Convert the state from a NumPy array to a PyTorch tensor
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        # Pass the state through the DQN to get action values
        action_values = self.dqn(state).detach().numpy().flatten()
        
        # Return the Q-value corresponding to the given action
        return action_values[action]

    #Maryem code: for task 4)b
    #
    def get_action_values(self, state):
        """
        Computes the Q-values for all actions in a given state using the Deep Q-Network (DQN).

        Parameters:
        state (numpy.ndarray): The current state of the environment represented as a NumPy array.

        Returns:
        numpy.ndarray: Array of Q-values for all actions in the given state.
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = self.dqn(state).numpy().flatten()
        return action_values


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


