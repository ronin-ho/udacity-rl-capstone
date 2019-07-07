from keras.layers import Layer
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import RMSprop
from agents.util import *

class RBFAgent(object):
    
    def __init__(self, env, beta=1):
        
        self.state_shape = env.observation_space.shape
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self.rbf_beta = beta
        
        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.3
        self.noise_epsilon = 0.99
        self.noise_decay_rate = 0.99999
        #self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        self.noise = NormalActionNoise(self.action_size, self.exploration_mu, 0.35)
        
        # Network
        model = Sequential() 
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape)) 
        model.add(Dense(16)) 
        model.add(Activation('relu'))
#        model.add(RBFLayer(self.state_size,betas=self.rbf_beta))
        model.add(RBFLayer(16,betas=self.rbf_beta))
#        model.add(RBFLayer(self.state_size, 
            #initializer=InitCentersRandom(X), 
#            betas=self.rbf_beta, 
#            input_shape=(1,) + self.state_shape))
                  
        model.add(Dense(self.action_size)) 
        model.compile(optimizer=RMSprop(), loss='mse')
        self.actor = model
        
        self.memory_state = []
        self.memory_action = []
        self.episode_reward = 0.
        
        self.best_episode_reward = -np.inf
        
    def act(self, state):
        
        state = [state]
        batch = np.array([state])
        action = self.actor.predict(batch)[0]
        
        #Noise decay
        noise = self.noise.sample() * self.noise_epsilon
        self.noise_epsilon = self.noise_epsilon * self.noise_decay_rate
        
        action = action + noise
        action = np.clip(action, self.action_low, self.action_high).astype(np.float32)
        
        return action
    
    def reset_episode(self):
        self.memory_state = []
        self.memory_action = []
        self.episode_reward = 0.
        #self.noise.reset()
        return

    def step(self, last_state, action, reward, next_state, done):
        
        self.memory_state.append(np.array([last_state]))
        self.memory_action.append(action)
        
        self.episode_reward += reward
       
        if done:
            self.learn()
        
        return
    
    def step_without_memory(self):
        return
    
    def learn(self):
        
        if self.episode_reward > self.best_episode_reward:
            self.best_episode_reward = self.episode_reward
            
            X = np.array(self.memory_state)
            y = np.array(self.memory_action)

            self.actor.fit(X, y,
              batch_size=40,
              epochs=1000,
              verbose=1)
        
        return

    def soft_update(self, local_model, target_model):
        return
       
    def normalize_states(self, states):
        return states
        
    def save_weight(self, path):
        return
        
    def load_weight(self, path):
        return
        