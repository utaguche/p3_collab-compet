import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    """
    initialize the weghts of a given layer
    """
    fan_in = layer.weight.data.size()[0]
    lim = np.sqrt(1.0/fan_in)

    return [-lim, lim]

class Actor(nn.Module):
    """
    Actor model
    that maps a state to a probability for taking actions.
    """

    def __init__(self, state_size, action_size, seed):
        """
        Initialization.
        Params
        =====
        state_size (int): state size
        action_size (int): action size
        seed (int): random seed
        """

        super(Actor, self).__init__()

        self.seed = torch.manual_seed(seed) # set the random seed
        self.state_size = state_size # set state size
        self.action_size = action_size # set action size
        self.hidden_units = [128,128] # assign the number of units for the hidden layers below

        # Linear layers
        self.fc1 = nn.Linear(self.state_size, self.hidden_units[0])
        self.fc2 = nn.Linear(self.hidden_units[0], self.hidden_units[1])
        self.fc3 = nn.Linear(self.hidden_units[1], self.action_size)

        self.reset_parameters() # initialize the weights

    def reset_parameters(self):

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Forwarding with relu and tanh

        """

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))

        return x



class Critic(nn.Module):
    """
    Critic network
    that maps (state, action) to Q-value
    """

    def __init__(self, state_size, action_size, num_agents, seed):
        """
        Initialization.

        Param
        =====
        state_size (int): state size
        action_size (int): action size
        num_agents (int): number of agents
        seed (int): random seed

        """

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed) # set the random seed
        self.state_size = state_size # set the state size
        self.action_size = action_size # set the action size

        # Need to centralize the states and actions of multiple agents
        self.state_size_tot = state_size * num_agents # set total state size
        self.action_size_tot = action_size * num_agents # set total action size

        self.hidden_units = [128 * num_agents, 128] # assign the number of units in the hidden layers below

        # Linear layers
        self.fc1 = nn.Linear(self.state_size_tot + self.action_size_tot, self.hidden_units[0])
        self.fc2 = nn.Linear(self.hidden_units[0], self.hidden_units[1])
        self.fc3 = nn.Linear(self.hidden_units[1], 1)

        self.reset_parameters() # initialize the weights

    def reset_parameters(self):
        """
        Initizlize the weights fitting into the extended number of units
        """

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state_tot, action_tot):
        """
        Forwarding with cat and relu

        Param:
        =====
        state__tot (tensor, float, dim=[batch_size, (state_size) * 2]):
          a tensor storing (flattened) states of 2 agents.
        action_tot (tensor, float, dim=[batch_size, (action_size)* 2]):
          a tensor storing (flattened) actions of 2 the agents.
        (output)
        - x (tensor, float, dim=[batch size, 1]): a tensor for Q-value
        """

        # Flatten both the total states and actions of 2 agents
        state_tot_flat = state_tot.view(-1, self.state_size_tot)
        action_tot_flat = action_tot.view(-1, self.action_size_tot)

        x = torch.cat([state_tot_flat, action_tot_flat], dim = 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x