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