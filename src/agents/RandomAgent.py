class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, env):
        self.action_space = env.action_space

    #def act(self, observation, reward, done):
    def act(self, state):
        return self.action_space.sample()
    
    def step(self, action, reward, next_state, done):
        return
    
    def reset_episode(self, state):
        return