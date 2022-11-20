import numpy as np
import torch
torch.manual_seed(0)
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from reinforce import optimize
nn_activation_function = nn.LeakyReLU()


class ConvLSTMContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, window_size=1, num_lstm_layers=1, action_bounds=1, dropout=0):
        super(ConvLSTMContinuous, self).__init__()
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = int(hidden_size/2)
        self.action_bounds = action_bounds
        self.conv_layer = nn.Conv1d(observation_space, hidden_size, kernel_size=window_size)
        self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=self.lstm_hidden_size, num_layers=self.num_lstm_layers, batch_first=True, dropout=dropout)
        self.mean_layer = nn.Linear(self.lstm_hidden_size, 1)        
        self.std_layer = nn.Linear(self.lstm_hidden_size, 1)

    def forward(self, x, hx=None) -> tuple[torch.Tensor, torch.Tensor]:
        #Conv layer
        x = x.unsqueeze(-1) 
        x = self.conv_layer(x)
        x = nn_activation_function(x)        
        x = x.squeeze().unsqueeze(0)
        if len(x.shape) == 3: #Batch TODO this looks bad
            x = x.squeeze()

        #LSTM layer
        if hx is not None:
            x, hx = self.lstm_layer(x, hx)
        else: 
            x, hx = self.lstm_layer(x)
        
        x = nn_activation_function(x)

        mean = torch.tanh(self.mean_layer(x))
        std = F.softplus(self.std_layer(x))

        return mean, std, hx#(hx, cx)
    
    def act(self, state, hx=None) -> tuple[float, torch.Tensor, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mean, std, hx = self.forward(state, hx)
        mean = torch.clamp(mean, -1, 1)
        std += 1e-5
        dist = Normal(mean, std) 
        action = dist.sample() 
        return torch.clamp(action, -1, 1).item(), dist.log_prob(action), hx


class LSTMContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, n_layers=1, dropout=0) -> None:
        super(LSTMContinuous, self).__init__()
        self.lstm_layer = nn.LSTM(input_size=observation_space, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.mean_layer = nn.Linear(hidden_size, 1)        
        self.std_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, hx=None) -> tuple[torch.Tensor, torch.Tensor]:
        if hx is not None:
            x, hx = self.lstm_layer(x, hx)
        else: 
            x, hx = self.lstm_layer(x)

        x = nn_activation_function(x)
        mean = torch.tanh(self.mean_layer(x))
        std = F.softplus(self.std_layer(x))
        return mean, std, hx
    
    def act(self, state, hx=None) -> tuple[float, torch.Tensor, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mean, std, hx = self.forward(state, hx)
        mean = torch.clamp(mean, -1, 1)
        std += 1e-5
        dist = Normal(mean, std) 
        action = dist.sample() 
        return torch.clamp(action, -1, 1).item(), dist.log_prob(action), hx


class LSTMDiscrete(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, n_layers=1, dropout=0) -> None:
        super(LSTMDiscrete, self).__init__()
        self.lstm_layer = nn.LSTM(input_size=observation_space, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_size, action_space)        

    def forward(self, x, hx=None) -> tuple[torch.Tensor, torch.Tensor]:
        if hx is not None:
            x, hx = self.lstm_layer(x, hx)
        else: 
            x, hx = self.lstm_layer(x)
        
        x = nn_activation_function(x)
        x = self.output_layer(x)
        return F.softmax(x, dim=1), hx
    
    def act(self, state, hx=None) -> tuple[float, torch.Tensor, tuple]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs, hx = self.forward(state, hx)  
        m = Categorical(probs) 
        action = m.sample() 
        return (action.item() - 1), m.log_prob(action), hx


def recurrent_reinforce(policy_network, env, alpha=1e-3, weight_decay=1e-5, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, train=True, print_res=True, print_freq=100) -> tuple[np.ndarray, np.ndarray]: 
    optimizer = optim.Adam(policy_network.parameters(), lr=alpha, weight_decay=weight_decay)
    reward_history = []
    action_history = []

    for n in range(num_episodes):
        state = env.reset() #S_0
        hx = None #h_0
        rewards = [] 
        actions = [] 
        log_probs = []  

        for _ in range(max_episode_length):
            action, log_prob, hx = policy_network.act(state, hx) #A_{t-1}
            state, reward, done, _ = env.step(action) # S_t, R_t 

            if done:
                break

            actions.append(action)
            rewards.append(reward) 
            log_probs.append(log_prob)

        if train:
            r = torch.FloatTensor(rewards)
            r = (r - r.mean()) / (r.std() + float(np.finfo(np.float32).eps))
            log_probs = torch.stack(log_probs).squeeze()
            policy_loss = torch.mul(log_probs, r).mul(-1).sum()
            optimize(optimizer, policy_loss)
        
        if print_res:
            if n % print_freq == 0:
                print("Episode ", n)
                print("Actions: ", np.array(actions))
                print("Sum rewards: ", sum(rewards))
                print("-"*20)
                print()
        
        reward_history.append(sum(rewards))
        action_history.append(np.array(actions))

    return np.array(reward_history), np.array(action_history)
