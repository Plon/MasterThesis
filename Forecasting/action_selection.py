import random
import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, MultivariateNormal
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# All functions are compatible with both recurrent and non-recurrent nets
# ------------------ One instrument 
## ----------------- Stochastic Sampling
### ---------------- Discrete Action Space (Multinoulli Dist)
# TODO maybe add random exploration to stochastic action selection
def act_stochastic_discrete(net: nn.Module, state: np.ndarray, hx=None, recurrent=False) -> tuple[float, torch.Tensor, tuple]:
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    if recurrent: 
        probs, hx = net.forward(state, hx)
    else: 
        probs = net.forward(state).cpu()
    probs = F.softmax(probs, dim=1)
    m = Categorical(probs) 
    action = m.sample() 
    return (action.item() - 1), m.log_prob(action), hx


### ---------------- Continuous Action Space (Gaussian Dist)
def act_stochastic_continuous(net: nn.Module, state: np.ndarray, hx=None, recurrent=False) -> tuple[float, torch.Tensor, torch.Tensor]:
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    if recurrent: 
        mean, std, hx = net.forward(state, hx)
    else: 
        mean, std = net.forward(state)
    mean = torch.clamp(mean, -1, 1)
    std = max(std, 0)
    std += 1e-8
    dist = Normal(mean, std) 
    action = dist.sample() 
    return torch.clamp(action, -1, 1).item(), dist.log_prob(action), hx


## ----------------- Deterministic Sampling 
### ---------------- Discrete Action Space (DQN)
def act_DQN(net: nn.Module, state: np.ndarray, hx=None, recurrent=False, epsilon=0) -> tuple[int, torch.Tensor]:
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        if recurrent: 
            action_vals, hx = net(state, hx)
        else: 
            action_vals = net(state).cpu()
    if random.random() < epsilon:
        action = np.random.randint(-1, 2) #[-1, 2)
    else:
        action = action_vals.argmax().item() - 1
    return action, hx


### ---------------- Continous Action Space (DDPG)
def act_DDPG(net: nn.Module, state: np.ndarray, hx=None, recurrent=False, epsilon=0, training=True) -> tuple[float, torch.Tensor]:
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        if recurrent:
            action, hx = net(state, hx)
        else: 
            action = net(state)
    action = torch.tanh(action)
    noise = ((np.random.rand(1)[0] * 2) - 1) #TODO maybe change to Ornstein-Uhlenbeck process
    action += training*max(epsilon, 0)*noise
    action = torch.clamp(action, -1, 1).item()
    return action, hx


# ------------------ Portfolio
## ----------------- Stochastic Sampling
### ---------------- Long & Short 
def act_stochastic_portfolio(net: nn.Module, state: np.ndarray, hx=None, recurrent=False, epsilon=0) -> tuple[float, torch.Tensor, tuple]:
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    if recurrent: 
        probs, hx = net.forward(state, hx)
    else: 
        probs = net.forward(state).cpu()
    m = MultivariateNormal(probs, torch.eye(probs.size(1))) # second arg is indentity matrix of size num instruments X num instruments
    action = m.sample()
    logprob = m.log_prob(action)
    action = torch.tanh(action)
    action = torch.clamp(action, -1, 1)
    action = np.array(action.squeeze())
    return action, logprob, hx


### ---------------- Long only softmax weighted (sum weights = 1)
def act_stochastic_portfolio_long(net: nn.Module, state: np.ndarray, hx=None, recurrent=False, epsilon=0) -> tuple[float, torch.Tensor, tuple]:
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    if recurrent: 
        probs, hx = net.forward(state, hx)
    else: 
        probs = net.forward(state).cpu()
    m = MultivariateNormal(probs, torch.eye(probs.size(1))) 
    action = m.sample()
    logprob = m.log_prob(action)
    action = F.softmax(action, dim=1)
    action = np.array(action.squeeze())
    return action, logprob, hx


## ----------------- Deterministic Sampling (DDPG)
### ---------------- Long & Short 
def act_DDPG_portfolio(net: nn.Module, state: np.ndarray, hx=None, recurrent=False, epsilon=0, training=True) -> tuple[float, torch.Tensor]:
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        if recurrent:
            action, hx = net(state, hx)
        else: 
            action = net(state)
    action = torch.tanh(action)
    noise = ((np.random.rand(1)[0] * 2) - 1) #TODO maybe change to Ornstein-Uhlenbeck process
    action += training*max(epsilon, 0)*noise
    action = torch.clamp(action, -1, 1)
    action = np.array(action.squeeze())
    return action, hx


### ---------------- Long only softmax weighted (sum weights = 1)
def act_DDPG_portfolio_long(net: nn.Module, state: np.ndarray, hx=None, recurrent=False, epsilon=0, training=True) -> tuple[float, torch.Tensor]:
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        if recurrent:
            action, hx = net(state, hx)
        else: 
            action = net(state)
    action = torch.tanh(action)
    #noise = ((np.random.rand(1)[0] * 2) - 1) #TODO maybe change to Ornstein-Uhlenbeck process
    #action += training*max(epsilon, 0)*noise
    action = F.softmax(action, dim=1)
    action = np.array(action.squeeze())
    return action, hx
