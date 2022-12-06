import torch
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
import yfinance as yf
from networks import AConvContinuous, AConvDiscrete, AConvLSTMContinuous, AConvLSTMDiscrete, AFFContinuous, FFDiscrete, ALSTMContinuous, ALSTMDiscrete, CConvSA, CFFSA, LinearDiscrete, LinearContinuous
from reinforce import reinforce
from deep_q_network import deep_q_network
from deep_deterministic_policy_gradient import deep_determinstic_policy_gradient
from reinforce_baseline import reinforce_baseline
import plotly.express as px
import matplotlib.pyplot as plt
from create_state_vector import get_states
from action_selection import act_stochastic_discrete, act_stochastic_continuous, act_DQN, act_DDPG
from proximal_policy_optimization import proximal_policy_optimization
from rl_portfolio_environment import PortfolioEnvironment

states, prices, _ = get_states(["CL=F"], imb_bars=True)
num_instruments = 1
num_features = states.shape[1]
num_prev_obs = 2
total_num_features = (num_features + num_instruments) * num_prev_obs

te = PortfolioEnvironment(states, num_instruments=num_instruments, num_prev_observations=num_prev_obs)

col_scores = []
num_runs = 500


### Reinforce Linear 
"""
policy = LinearDiscrete(observation_space=total_num_features).to(device)
scores, actions = reinforce(policy, te, act=act_stochastic_discrete, num_episodes=1000)
#"""

### REINFORCE linear CONTINUOUS ACTION SPACE
"""
policy = LinearContinuous(observation_space=total_num_features, action_space=1).to(device)
scores, actions = reinforce(policy, te, act=act_stochastic_continuous, alpha=1e-3, num_episodes=2001)
#"""


### REINFORCE feedforward
"""
policy = FFDiscrete(observation_space=total_num_features).to(device)
scores, actions = reinforce(policy, te, act=act_stochastic_discrete, num_episodes=400)
#"""

### REINFORCE feedwordward with baseline
"""
policy = FFDiscrete(observation_space=total_num_features).to(device)
value_function = FFDiscrete(observation_space=total_num_features, action_space=1).to(device)
scores, actions = reinforce_baseline(policy, value_function, te, act=act_stochastic_discrete,  alpha_policy=1e-3, alpha_vf=1e-5)
#"""

### REINFORCE feedforward CONTINUOUS ACTION SPACE
"""
policy = AFFContinuous(observation_space=total_num_features).to(device)
scores, actions = reinforce(policy, te, act=act_stochastic_continuous, alpha=1e-4)
#"""

### REINFORCE conv with baseline
"""
policy = AConvDiscrete(observation_space=total_num_features).to(device)
value_function = AConvDiscrete(observation_space=total_num_features, action_space=1).to(device)
#value_function = FFDiscrete(observation_space=total_num_features, action_space=1).to(device)
scores, actions = reinforce_baseline(policy, value_function, te, act=act_stochastic_discrete,alpha_policy=1e-4, alpha_vf=1e-5)
#"""

### Recurrent REINFORCE LSTM Discrete  Action Space
"""
policy = ALSTMDiscrete(observation_space=total_num_features, n_layers=2, dropout=0.1).to(device)
scores, actions = reinforce(policy, te, alpha=1e-4, act=act_stochastic_discrete, recurrent=True)
#"""

### Recurrent REINFORCE LSTM Continuous Action Space
"""
policy = ALSTMContinuous(observation_space=total_num_features, n_layers=2, dropout=0.1).to(device)
scores, actions = reinforce(policy, te, alpha=1e-4, act=act_stochastic_continuous, recurrent=True)
#"""

### Recurrent REINFORCE with baseline LSTM Discrete Action Space
"""
policy = ALSTMDiscrete(observation_space=total_num_features, n_layers=2, dropout=0.1).to(device)
value_function = FFDiscrete(observation_space=total_num_features, action_space=1).to(device)
scores, actions = reinforce_baseline(policy, value_function, te, act=act_stochastic_discrete, alpha_policy=1e-4, alpha_vf=1e-5, recurrent=True)
#"""

### Recurrent REINFORCE with baseline LSTM Continuous Action Space
"""
policy = ALSTMContinuous(observation_space=total_num_features, n_layers=2, dropout=0.1).to(device)
value_function = FFDiscrete(observation_space=total_num_features, action_space=1).to(device)
scores, actions = reinforce_baseline(policy, value_function, te, act=act_stochastic_continuous, alpha_policy=1e-4, alpha_vf=1e-5, recurrent=True)
#"""

### REINFORCE Conv
"""
policy = AConvDiscrete(observation_space=total_num_features).to(device)
scores, actions = reinforce(policy, te, act=act_stochastic_discrete, alpha=1e-4)
#"""

### REINFORCE Conv Continuous Action Space
"""
policy = AConvContinuous(observation_space=total_num_features).to(device)
scores, actions = reinforce(policy, te, act=act_stochastic_continuous, alpha=1e-5)
#"""

### Recurrent REINFORCE Conv LSTM 
"""
policy = AConvLSTMDiscrete(observation_space=total_num_features, num_lstm_layers=2).to(device)
scores, actions = reinforce(policy, te, act=act_stochastic_discrete, alpha=1e-4, recurrent=True, print_freq=1)
#"""

### Recurrent REINFORCE Conv LSTM Continuous Action Space
"""
policy = AConvLSTMContinuous(observation_space=total_num_features, num_lstm_layers=2).to(device)
scores, actions = reinforce(policy, te, act=act_stochastic_continuous, alpha=1e-4, recurrent=True, print_freq=1)
#"""

### DQN 
"""
q_net = FFDiscrete(observation_space=total_num_features, action_space=3).to(device)
#q_net = AConvDiscrete(observation_space=total_num_features, action_space=3).to(device)
scores, actions = deep_q_network(q_net, te, act=act_DQN, batch_size=64, alpha=1e-4, num_episodes=501)
#"""

### DRQN 
"""
#q_net = AConvLSTMDiscrete(observation_space=total_num_features, action_space=3, num_lstm_layers=1).to(device)
q_net = ALSTMDiscrete(observation_space=total_num_features, action_space=3, n_layers=2).to(device)
scores, actions = deep_q_network(q_net, te, act=act_DQN, batch_size=64, alpha=1e-4, num_episodes=501, recurrent=True)
#"""

### DDPG
#"""
#actor = AConvLSTMDiscrete(observation_space=total_num_features, action_space=1).to(device)
#actor = ALSTMDiscrete(observation_space=total_num_features, action_space=1, n_layers=2).to(device)
#actor = AConvDiscrete(observation_space=total_num_features, action_space=1).to(device)
actor = FFDiscrete(observation_space=total_num_features, action_space=1).to(device)
#critic = CConvSA(observation_space=total_num_features).to(device)
critic = CFFSA(observation_space=total_num_features).to(device)
scores, actions = deep_determinstic_policy_gradient(actor, critic, te, act=act_DDPG, batch_size=128, alpha_actor=1e-4, alpha_critic=1e-3, num_episodes=401, recurrent=False)
#"""

### PPO Linear - idea doesnt fit the problem at all
"""
actor = FFDiscrete(observation_space=total_num_features).to(device)
#critic = LinearDiscrete(observation_space=total_num_features, action_space=3).to(device)
critic = CFFSA(observation_space=total_num_features, action_space=1).to(device)
scores, actions = proximal_policy_optimization(actor, critic, te, act=act_stochastic_discrete, batch_size=32, alpha_actor=1e-3, alpha_critic=1e-5, num_episodes=1001, recurrent=False)
#"""

### PLOTS
"""
labels = ["R", "R_CA", "R_LSTM", "R_LSTM_CA", "R_Conv", "R_Conv_CA", "AC", "AC_CA"]
for s, l in zip(col_scores, labels):
    plt.plot(s, label=l)
plt.xlabel("Run_nr")
plt.ylabel("{%} return")
plt.legend()
plt.show()
#"""

with np.printoptions(threshold=np.inf):
    print(actions[-1])

correct_actions = np.sign(prices[0][1:] - prices[0][:-1])
print(correct_actions)
print(correct_actions == actions[-1])
print(sum(correct_actions == actions[-1]) - len(correct_actions))

retrn = 0
optimal_rertn = 0
prev_p = prices[0][0]
prev_a = 0
prev_a_optimal = 0
for a, p, oa in zip(actions[-1], prices[0][1:], correct_actions):
    retrn += a * (p - prev_p) - (prev_p * 0.002 * abs(a - prev_a))
    optimal_rertn += oa * (p - prev_p) - (prev_p * 0.002 * abs(oa - prev_a_optimal))
    prev_p = p
    prev_a = a
    prev_a_optimal = oa


print(retrn, optimal_rertn)


#"""
#fig = px.line(x=np.linspace(1, len(scores), num=len(scores)), y=scores)    
#fig.show()

from scipy.signal import savgol_filter
fig = px.line(x=np.linspace(1, len(scores), num=len(scores)), y=savgol_filter(scores, 9, 2))
fig.show()

#fig = px.line(x=np.linspace(1, len(scores), num=len(scores)), y=np.cumsum(scores))
#fig.show()
#"""