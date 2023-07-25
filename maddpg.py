import numpy as np
import copy
from collections import deque, namedtuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

# Hyper parameters
BUFFER_SIZE = 100000 # the size of the replay buffer
BATCH_SIZE = 256 # batch size
GAMMA = 0.999 # discount factor
TAU =  0.1 # parameter for soft update

LEARN_EVERY = 1 # time steps per which the learning occurs
LR_ACTOR = 0.0001 # learning rate for actor model
LR_CRITIC = 0.001 # learning rate for critic model

COUNT_LEARNING = 1 # frequency per which the updates occur per learning
WEIGHT_DECAY_ACTOR = 0.0 # weight decay for the actor
WEIGHT_DECAY_CRITIC = 0.0 # weight decay for the critic

EPSILON = 1.0 # initial noise factor
EPSILON_DECAY = 0.999 # noise decay rate
EPSILON_MIN = 0.3 # lower bound for epsilon

# Select the device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPG_Agent():
    """
    DDPG agent which interacts with and learns from the environment.
    """

    def __init__(self, state_size, action_size, num_agents, seed):
        """
        Initialization.
        Params
        =====
        state_size (int): state size for a single agent
        size_action (int): action size for a single agent
        num_agents (int): number of agents
        seed (int): random seed
        """

        self.seed = torch.manual_seed(seed) # set random seed
        self.action_size = action_size # set action size
        self.state_size = state_size # set state size
        self.num_agents = num_agents # set number of agents

        # Actor networks (local+target)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_ACTOR, weight_decay = WEIGHT_DECAY_ACTOR)

        # Critic networks (local+target)
        self.critic_local = Critic(state_size, action_size, num_agents, seed).to(device)
        self.critic_target = Critic(state_size, action_size, num_agents, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_CRITIC, weight_decay = WEIGHT_DECAY_CRITIC)

        # Initial soft updates
        self.soft_update(self.actor_local, self.actor_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)

        # Noise process
        self.noise = OUNoise(action_size, seed)
        self.epsilon = EPSILON

    def act(self, state, add_noise = True):
        """
        Select an action for a given state as per policy and add noise.

        """

        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # Generate noise for the selected action
        if add_noise == True:
            noise_sample = self.noise.sample() * self.epsilon
            action += noise_sample

        # Update the noise factor not to go below the lower bound
        self.epsilon = max(self.epsilon*EPSILON_DECAY, EPSILON_MIN)

        return np.clip(action, -1, 1)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """

        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class MADDPG_Agent():
    """
    MADDPG agent
    """

    def __init__(self, state_size, action_size, num_agents, seed):
        """
        Initialization
        Params
        =====
        state_size (int): state size for a single DDPG agent
        action_size (int): action size for a single DDPG agent
        num_agents (int): number of agents
        seed (int): random seed
        """

        self.seed = torch.manual_seed(seed) # set random seed
        self.state_size = state_size # set state size
        self.action_size = action_size # set action size
        self.num_agents = num_agents # set number of agents

        # Create DDPG agents in a list
        self.maddpg_agents = [DDPG_Agent(state_size, action_size, num_agents, seed) for _ in range(num_agents)]

        # Replay buffer
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

    def reset(self):
        # Reset time step for learning per LEARN_EVERY
        self.time_stamp = 0

        # Reset the noise
        for i in range(self.num_agents):
            self.maddpg_agents[i].noise.reset()

    def act_all(self, state_tot, add_noise = True):
        """
        All the agents select the actions as per policy.

        Params
        =====
        state_tot (numpy array, float, dim = [batch_size, 2, state_size]:
            total states for two DDPG agents
        """

        action_tot = []

        # if avoiding slicing
        for state, ddpg_agent in zip(state_tot, self.maddpg_agents):
            action = ddpg_agent.act(state, add_noise)
            action_tot.append(action)

        return action_tot

    def step(self, state_tot, action_tot, reward_tot, next_state_tot, done_tot):
        """
        Store a tuple of experiences in replay buffer and run the learning process.
        (input)
        state_tot (tensor, float, dim = [2, state_size])
        action_tot (tensor, float, dim = [2, action size])
        reward_tot (tensor, float, dim = [2, 1])
        next_state_tot (tensor, float, dim = [2, state size])
        done_tot (tensor, int, dim = [2, 1])
        """

        # Save a tuple of experiences to the replay buffer
        self.memory.add(state_tot, action_tot, reward_tot, next_state_tot, done_tot)

        # Take the learning process COUNT_LEARNING times per LEARN_EVERY
        self.time_stamp  = (self.time_stamp + 1) % LEARN_EVERY
        if self.time_stamp == 0:
            if len(self.memory) > BATCH_SIZE:
                for _ in range (COUNT_LEARNING):
                    for i_agent in range(self.num_agents):
                        self.learn(i_agent, GAMMA)
                    # Soft update the target networks
                    self.soft_update_all(TAU)

    def learn(self, i_agent, gamma):
        """
        Learning process using given batch of experience tuples.
        Q_target = rewards + gamma * Q_target_next * (1 - dones)
        Params
        =====
        i_agent (int): the index for the DDPG agents
        gamma (float): discount factor
        """

        # Take experiences from the replay buffer
        experiences = self.memory.sample(i_agent)
        states, states_tot, actions, actions_tot, rewards, next_states, next_states_tot, dones = experiences

        # Send the data to the device
        states = states.to(device)
        states_tot = states_tot.to(device)
        actions = actions.to(device)
        actions_tot = actions_tot.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        next_states_tot = next_states_tot.to(device)
        dones = dones.to(device)

        # Update critic
        # Get the predicted next actions and Q values from target models
        next_actions_tot = []
        for i in range(self.num_agents):
            next_state_i =  next_states_tot[:,i,:]
            next_action_i = self.maddpg_agents[i].actor_target(next_state_i)
            next_actions_tot.append(next_action_i)
        next_actions_tot = torch.cat(next_actions_tot, dim=1).to(device)
        # Compute Q target for the current states
        Q_target_next = self.maddpg_agents[i_agent].critic_target(next_states_tot, next_actions_tot)
        Q_target = rewards + gamma * Q_target_next * (1 - dones)
        # Compute critic loss and back propagate
        Q_expected = self.maddpg_agents[i_agent].critic_local(states_tot, actions_tot)
        critic_loss = F.mse_loss(Q_expected, Q_target)
        # Back propagation for the critic
        self.maddpg_agents[i_agent].critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip the gradient
        nn.utils.clip_grad_norm_(self.maddpg_agents[i_agent].critic_local.parameters(), 1.0) # gradient clipping
        self.maddpg_agents[i_agent].critic_optimizer.step()

        # Update actor
        # Compute actor loss
        actions_pred = self.maddpg_agents[i_agent].actor_local(states)
        actions_pred_tot = [actions_tot[:, i, :] if i != i_agent else actions_pred for i in range(self.num_agents)]
        actions_pred_tot = torch.cat(actions_pred_tot, dim = 1).to(device)
        # Note the minus sign!
        actor_loss = - self.maddpg_agents[i_agent].critic_local(states_tot, actions_pred_tot).mean()
        # Back propagation for the actor
        self.maddpg_agents[i_agent].actor_optimizer.zero_grad()
        actor_loss.backward()
        self.maddpg_agents[i_agent].actor_optimizer.step()


    def soft_update_all(self, tau):
        """
        Soft update for all the target models.

        """

        for i in range(self.num_agents):
            self.maddpg_agents[i].soft_update(self.maddpg_agents[i].actor_local, self.maddpg_agents[i].actor_target, tau)
            self.maddpg_agents[i].soft_update(self.maddpg_agents[i].critic_local, self.maddpg_agents[i].critic_target, tau)

class OUNoise():
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu = 0.0, theta = 0.15, sigma = 0.1):
        """Initializate parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed = 27):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.seed = random.seed(seed)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(maxlen = self.buffer_size)
        self.experience = namedtuple("Experience", field_names = ['state_tot', 'action_tot', 'reward_tot', 'next_state_tot', 'done_tot'])

    def add(self, state_tot, action_tot, reward_tot, next_state_tot, done_tot):
        """Add a new experience to replay buffer"""
        e = self.experience(state_tot, action_tot, reward_tot, next_state_tot, done_tot)
        self.memory.append(e)

    def sample(self, i_agent):
        """Random-sample a batch of experiences from memory"""

        experiences = random.sample(self.memory, k = self.batch_size)

        states = torch.from_numpy(np.vstack([e.state_tot[i_agent] for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action_tot[i_agent] for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward_tot[i_agent] for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state_tot[i_agent] for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done_tot[i_agent] for e in experiences if e is not None]).astype(np.uint8)).float()
        states_tot = torch.from_numpy(np.vstack([[e.state_tot] for e in experiences if e is not None])).float()
        actions_tot = torch.from_numpy(np.vstack([[e.action_tot] for e in experiences if e is not None])).float()
        next_states_tot = torch.from_numpy(np.vstack([[e.next_state_tot] for e in experiences if e is not None])).float()

        return (states, states_tot, actions, actions_tot, rewards, next_states, next_states_tot, dones)

    def __len__(self):
        """Return the current size of the replay buffer memory"""
        return len(self.memory)