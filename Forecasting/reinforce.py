import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PolicyNetworkContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=1, action_bounds=1) -> None:
        super(PolicyNetworkContinuous, self).__init__()
        self.action_bounds = action_bounds
        self.input_layer = nn.Linear(observation_space, hidden_size)
        self.output_layer = nn.Linear(hidden_size, action_space)

        logstds_param = nn.Parameter(torch.full((action_space,), 0.1))
        self.register_parameter("logstds", logstds_param)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.input_layer(x))
        return self.output_layer(x)
    
    def act(self, state) -> tuple[float, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mean = self.forward(state).cpu()
        std = torch.clamp(self.logstds.exp(), 1e-3, 50)
        dist = Normal(mean, std) 
        action = dist.sample() 
        return torch.clamp(action, -self.action_bounds, self.action_bounds).item(), dist.log_prob(action)


class PolicyNetworkDiscrete(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3) -> None:
        super(PolicyNetworkDiscrete, self).__init__()
        self.input_layer = nn.Linear(observation_space, hidden_size)
        self.output_layer = nn.Linear(hidden_size, action_space)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.input_layer(x))
        x = self.output_layer(x)
        return F.softmax(x, dim=1)
    
    def act(self, state) -> tuple[int, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()        
        m = Categorical(probs) 
        action = m.sample() 
        return (action.item() - 1), m.log_prob(action)


def train_policy_network(delta, log_prob, optimizer) -> None: 
    """ Calculate loss for policy and backpropagate """
    loss = -log_prob * delta
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def reinforce(policy_network, env, alpha=1e-3, num_episodes=np.iinfo(np.int32).max) -> tuple[np.ndarray, np.ndarray]: 
    """
    Online Monte Carlo policy gradient 
    Every trajectory consists of one step, and then the loss is computed

    Args: 
        policy_network (nn): network that parameterizes the policy
        env: environment that the rl agent interacts with
        num_episodes (int): maximum number of episodes
    Returns: 
        scores (numpy.ndarray): the rewards of each episode
        actions (numpy.ndarray): the actions chosen by the agent
    """
    optimizer = optim.Adam(policy_network.parameters(), lr=alpha)
    scores = [] 
    actions = [] 
    state = env.state() #S_0

    for _ in range(num_episodes):
        action, log_prob = policy_network.act(state) #A_{t-1}
        state, reward, done, _ = env.step(action) # S_t, R_t 

        if done:
            break

        actions.append(action)
        scores.append(reward) 
        train_policy_network(reward, log_prob, optimizer)

    return np.array(scores), np.array(actions)
