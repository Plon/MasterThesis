
# takes a n number of instruments as input
# returns softmax weights of n instruments -> no short 
# for the moment optimizes for sharpe
# will probably have to change some of the rl algos

# should a unique neural net output one likelihood per instrument and then softmax on all, 
# or should it output all the outputs from one net?

import torch
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
from networks import AConvContinuous, AConvDiscrete, AConvLSTMContinuous, AConvLSTMDiscrete, AFFContinuous, FFDiscrete, ALSTMContinuous, ALSTMDiscrete, CConvSA, CFFSA
from reinforce import reinforce
from deep_q_network import deep_q_network
from deep_deterministic_policy_gradient import deep_determinstic_policy_gradient
from reinforce_baseline import reinforce_baseline
import plotly.express as px
import matplotlib.pyplot as plt
from rl_portfolio_environment import PortfolioEnvironment
from create_state_vector import get_states
from action_selection import act_stochastic_portfolio

#instruments = ["CL=F", "NG=F"] #WTI crude futures, Natural gas futures
instruments = ["CL=F", "NG=F", "ALI=F"] #WTI crude futures, Natural gas futures, Aluminium futures
num_instruments = len(instruments) + 1 # +1 if riskfree asset
period = "30d"
interval = "30m"
states, prices, _ = get_states(instruments, period, interval, imb_bars=False, riskfree_asset=True)

num_features = states.shape[1]
num_prev_obs = 2
total_num_features = (num_features + num_instruments) * num_prev_obs

pe = PortfolioEnvironment(states, num_instruments=num_instruments, num_prev_observations=num_prev_obs)


# TODO add compatibility to DDPG - error calculation - will not work because the actor loss takes tanh of actions
# TODO add compatibility to DQN - error calculation - will only work with discrete action space and will not scale for many instruments. a lost cause to implement
# TODO try to make rl_portfolio_env compatible with single instrument trading

### REINFORCE feedforward
#"""
policy = FFDiscrete(observation_space=total_num_features, action_space=num_instruments).to(device)
#policy = AConvDiscrete(observation_space=total_num_features, action_space=num_instruments).to(device)
scores, actions = reinforce(policy, pe, act=act_stochastic_portfolio, alpha=1e-3, num_episodes=401)
#"""

### Recurrent REINFORCE feedforward
"""
#policy = FFDiscrete(observation_space=total_num_features, action_space=num_instruments).to(device)
policy = AConvDiscrete(observation_space=total_num_features, action_space=num_instruments).to(device)
scores, actions = reinforce(policy, pe, act=act_stochastic_portfolio, alpha=1e-3, num_episodes=400, print_freq=1)
#"""

### REINFORCE feedwordward with baseline
"""
policy = FFDiscrete(observation_space=total_num_features, action_space=num_instruments).to(device)
#policy = AConvDiscrete(observation_space=total_num_features, action_space=num_instruments).to(device)
value_function = FFDiscrete(observation_space=total_num_features, action_space=1).to(device)
scores, actions = reinforce_baseline(policy, value_function, pe, act=act_stochastic_portfolio,  alpha_policy=1e-3, alpha_vf=1e-5, num_episodes=3000)
#"""


### PLOTS
from scipy.signal import savgol_filter
fig = px.line(x=np.linspace(1, len(scores), num=len(scores)), y=savgol_filter(scores, 9, 2))
fig.show()

