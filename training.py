"""
Add hyperparameters to argparse there.
- environment to test
- batchsize
- replay memory size
- update frequency
- discount factor
- learning rate
- gradient momentum
- initial exploration
- final exploration
- final exploration frame
- no-op max

- parser = argparse.ArgumentParser(description="Evaluate a Random Agent in a Gymnasium environment.")
    # Define the argument for the Gym environment name
    parser.add_argument("--env", type=str, required=True, help="The Gymnasium environment to use.")

    #*****extend_auf4b*****
    parser.add_argument("--agent", type=str, choices=["random", "dqn"], default="random", help="The type of agent to use.")
    #**********************

    # Define the argument for the number of episodes
    parser.add_argument("--episodes", type=int, default=10, help="The number of episodes to run.")
    # Parse the command-line arguments

    parser.add_argument("--log_wandb", action="store_true", help="Log results to Weights & Biases.")



Follow your paper for appropriate values.
Write your code as efficient as possible since training can take hours to days. Consider tricks
from pytorch optimization.ipynb (in our GitHub repository) for your code. Start to
integrate them when your algorithm starts working. Some suggested improvements might
not increase the speed, so benchmark them.
Implement a time limit and environment step limit to terminate your code.




"""
import argparse
import random
from collections import deque
import numpy as np

class ReplayMemory():

    def __init__(self, args):
        self.args = args
        self.memory = deque(maxlen=args.replay)

    def save_exp(self, exp):
        if len(self.memory) != self.args.replay:
            self.memory.append(exp)
        else:
            self.summarizeReplays()
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

        aggregated_experience = (agg_state, agg_action, agg_reward, agg_next_state)
        self.memory.appendleft(aggregated_experience)

    def get_exp(self):
        return random.sample(self.memory, self.args.batch)

class NetworkTraining():

    def __init__(self, args):
        self.args = args
        self.replay_memory = ReplayMemory(args)


    def training(self):
        print("übung macht den meister")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Random Agent in a Gymnasium environment.")
    parser.add_argument("--env", type=str, help="The environment to learn")
    parser.add_argument("--batch", type=int, default=32, help="size of the minibatch. number of replays in it.")
    parser.add_argument("--replay", type=int, default=1000000, help="memory size, storage of replays")
    parser.add_argument("--update_freq", type=int, default=10000, help="Log results to Weights & Biases.")
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

    training = NetworkTraining(args=args)
