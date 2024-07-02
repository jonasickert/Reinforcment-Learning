import gymnasium as gym

class random_agent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, state):
        return self.action_space.sample()



# A little test 
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