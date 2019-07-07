class RandomAgent(object):
    
    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, state):
        return self.action_space.sample()
    
    def reset_episode(self):
        return

    def step(self, last_state, action, reward, next_state, done):
        return
    
    def step_without_memory(self):
        return
    
    def learn(self, experiences):
        return

    def soft_update(self, local_model, target_model):
        return
       
    def normalize_states(self, states):
        return states
        
    def save_weight(self, path):
        return
        
    def load_weight(self, path):
        return
        