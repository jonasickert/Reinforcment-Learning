import networks
import gymnasium as gym
import torch
import io
import numpy as np

class DQNAgent():

    #code von Jonas
    #Korrektur von Maryem: removed len and .shape to test on CartPole-v1
    def __init__(self, env:gym.Env, input_dim, filepath):
        """
        Initializes the DQNAgent with the given environment and input dimension.
        Parameters:
        env (gym.Env): The environment in which the agent will operate.
        input_dim (int): The dimension of the input (observation space).
        """
        # initialisiere Netzwerk
        self.dqn = networks.QFunction(input_dim=input_dim, output_dim=env.action_space.n, input_type="pixels")
        if filepath is not None:
            self.load_network(filepath)

    #code von Maryem:
    def select_action(self, state):
        """
        Selects an action based on the given state using the Deep Q-Network (DQN).

        Parameters:
        state (numpy.ndarray): The current state of the environment represented as a NumPy array.

        Returns: the action the agent has to choose
        int: The selected action, represented as an integer.
        """
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0)  # Convert state to tensor
        # Disable gradient calculation since we are in evaluation mode
        with torch.no_grad():
            action_values = self.dqn(state)
        
        # Select the action with the highest value
        action = torch.argmax(action_values, dim=1).item()
        
        return action

    #code von Maryem for task 4)b
    def compute_action_value(self, state, action):
        """
        Computes the Q-value for a given state-action pair using the Deep Q-Network (DQN).

        Parameters:
        state (numpy.ndarray): The current state of the environment represented as a NumPy array.
        action (int): The action for which to compute the Q-value.

        Returns: action value of given state and action
        float: The Q-value for the given state-action pair.
        """

        # Convert the state from a NumPy array to a PyTorch tensor
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0)

        
        # Pass the state through the DQN to get action values
        action_values = self.dqn(state).detach().numpy().flatten()
        
        # Return the Q-value corresponding to the given action
        return action_values[action]

    
    #jonas: def save_network(self, name):
        """ saves the network parameters as name
        :param name: of the save
        """
    #jonas:    torch.save(self.dqn.state_dict(), name)

    
    #jonas:def load_network(self, name):
        """ load the network
        :param name: of the loaded network
        """
    #jonas:    torch.load(self.dqn.load_state_dict(torch.load(name)))

    #Maryem :
    def save_network(self, file_path):
        torch.save(self.dqn.state_dict(), file_path)

    def load_network(self, file_path):
        if isinstance(file_path, dict):
            buffer = io.BytesIO()
            torch.save(file_path, buffer)
            buffer.seek(0)
            self.dqn.load_state_dict(torch.load(buffer))
        else:
            self.dqn.load_state_dict(torch.load(file_path))

"""
Extend your agent class with functionality for training the networks. 
This is the implementation of the RL algorithm and is the main part of this project.


we need the stuff like in jupyter lab:
- predict

- sollen wir in DQAAgent das training ausführen?

"""


