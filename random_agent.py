import gymnasium as gym

class random_agent:

    """
    Agent that selects actions randomly from the action space.

    Attributes:
        action_space (gym.Space): Action space of the environment.
    """

    def __init__(self, action_space):
        """
        Initialize the agent.

        Args:
            action_space (gym.Space): Action space of the environment.
        """
        self.action_space = action_space

    def select_action(self, state):
        """
        Select an action randomly from the action space.

        Args:
            state: Current state of the environment.
        Returns:
            int: Randomly selected action index.
        """
        return self.action_space.sample()



#A little test
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = random_agent(env.action_space)
    state, _ = env.reset()
    done = False
    step = 0

    while not done:
        action = agent.select_action(state)
        state, reward, done, truncated, info = env.step(action)
        print(f"Step {step}: Selected action: {action}")
        step += 1