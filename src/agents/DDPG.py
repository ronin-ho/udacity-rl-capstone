import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
import random
from collections import namedtuple, deque
import copy

from agents.util import *

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_shape = task.observation_space.shape
        self.state_size = task.observation_space.shape[0]
        self.action_size = task.action_space.shape[0]
        self.action_low = task.action_space.low
        self.action_high = task.action_space.high

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.1
        self.noise_epsilon = 0.99
        self.noise_decay_rate = 0.999998
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 1000000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters

        # Learning Rate
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        
        self.use_normalizer = False
        self.state_normalizer = None


    def build_actor(self):
        state_input = Input(shape=(1,) + self.state_shape, name='state_input')
        state_net = Flatten()(state_input)
        state_net = Dense(400)(state_net)
        state_net = Activation('relu')(state_net)
        state_net = Dense(300)(state_net)
        state_net = Activation('relu')(state_net)
        state_net = Dense(self.action_size)(state_net)
        state_net = Activation('tanh')(state_net)
        actor = Model(inputs=[state_input], outputs=state_net)
        return actor
        
    def build_critic(self):
        action_input = Input(shape=(self.action_size,), name='action_input')
        state_input = Input(shape=(1,) + self.state_shape, name='state_input')
        net = Flatten()(state_input)
        net = Dense(400)(net)
        net = Activation('relu')(net)
        net = Concatenate()([net, action_input])
        net = Dense(300)(net)
        net = Activation('relu')(net)
        net = Dense(1)(net)
        net = Activation('linear')(net)
        critic = Model(inputs=[state_input, action_input], outputs=net)
        return critic

    def build_models(self):
        # Set up target network
        self.actor_local = self.build_actor()
        self.actor_local_optimizer = optimizers.Adam(lr=self.actor_lr)
        self.actor_local.compile(optimizer=self.actor_local_optimizer, loss='mse')

        self.actor_target = self.build_actor()
        self.actor_target_optimizer = optimizers.Adam(lr=self.actor_lr)
        self.actor_target.compile(optimizer=self.actor_target_optimizer, loss='mse')

        self.critic_local = self.build_critic()
        self.critic_local_optimizer = optimizers.Adam(lr=self.critic_lr)
        self.critic_local.compile(optimizer=self.critic_local_optimizer, loss='mse')

        self.critic_target = self.build_critic()
        self.critic_target_optimizer = optimizers.Adam(lr=self.critic_lr)
        self.critic_target.compile(optimizer=self.critic_target_optimizer, loss='mse')

        # Copy model weight
        self.critic_target.set_weights(self.critic_local.get_weights())
        self.actor_target.set_weights(self.actor_local.get_weights())

        state_inputs = Input(shape=(1,) + self.state_shape, name='state_inputs')
        actions = self.actor_local([state_inputs])
        q_values = self.critic_local([state_inputs, actions])
        updates = self.actor_local_optimizer.get_updates(params=self.actor_local.trainable_weights, loss=-K.mean(q_values))
        self.actor_train_fn = K.function([state_inputs] + [K.learning_phase()], [self.actor_local(state_inputs)], updates=updates)
        
        
    def reset_episode(self):
        self.noise.reset()

    def step(self, last_state, action, reward, next_state, done):

        # Save experience / reward
        self.memory.add(last_state, action, reward, next_state, done)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
    
    def step_without_memory(self):

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
    
    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        
        state = [state]
        batch = np.array([state])
        normalized_state = self.normalize_states(batch)
        
        action = self.actor_local.predict(normalized_state)[0]
        
        #Noise decay
        noise = self.noise.sample() * self.noise_epsilon
        self.noise_epsilon = self.noise_epsilon * self.noise_decay_rate
        
        if K.learning_phase():
            action = action + noise
            
        action = np.clip(action, self.action_low, self.action_high).astype(np.float32)
        return action

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        
        for e in experiences:
            states.append(np.array([e.state]))
            actions.append(e.action)
            rewards.append(e.reward)
            dones.append(e.done)
            next_states.append(np.array([e.next_state]))
            
        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        actions = np.array(actions)
        dones = np.array(dones)
        
        # Noramlize the states
        states = self.normalize_states(states)
        next_states = self.normalize_states(next_states)
        
        assert actions.shape == (self.batch_size, self.action_size)
        
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.predict_on_batch([next_states, actions_next]).flatten()

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        Q_targets = Q_targets.reshape(self.batch_size, 1)
        self.critic_local.train_on_batch(x=[states, actions], y=Q_targets)

        # Execute custom training function to train actor network
        self.actor_train_fn([states] + [1])
        
        # Soft-update target models
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
       
    def normalize_states(self, states):
        if self.use_normalizer:
            if self.state_normalizer is None:
                self.state_normalizer = WhiteningNormalizer(shape=states.shape[1:], dtype=states.dtype)
            self.state_normalizer.update(states)
            return self.state_normalizer.normalize(states)
        else:
            return states
        
    def save_weight(self, path):
        self.critic_local.save_weights(path + 'critic_local.h5')
        self.actor_local.save_weights(path + 'actor_local.h5')
        self.critic_target.save_weights(path + 'critic_target.h5')
        self.actor_target.save_weights(path + 'actor_target.h5')
        
        self.critic_local.save(path + 'critic_local_model.h5')
        self.actor_local.save(path + 'actor_local_model.h5')
        self.critic_target.save(path + 'critic_target_model.h5')
        self.actor_target.save(path + 'actor_target_model.h5')
        
    def load_weight(self, path):
        self.critic_local.load_weights(path + 'critic_local.h5')
        self.actor_local.load_weights(path + 'actor_local.h5')
        self.critic_target.load_weights(path + 'critic_target.h5')
        self.actor_target.load_weights(path + 'actor_target.h5')
        
    def load_model(self, path):
        self.critic_local = load_model(path + 'critic_local_model.h5')
        self.actor_local = load_model(path + 'actor_local_model.h5')
        self.critic_target = load_model(path + 'critic_target_model.h5')
        self.actor_target = load_model(path + 'actor_target_model.h5')

        
        