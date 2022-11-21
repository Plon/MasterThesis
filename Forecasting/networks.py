# Neural nets that output a position on the continuous interval [-1, 1] or {-1, 0, 1}
import random
import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.autograd import Variable 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#nn_activation_function = nn.ReLU()
nn_activation_function = nn.LeakyReLU()
#nn_activation_function = nn.ELU()


def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)


## ----------------- Convolutional + LSTM
### ---------------- Discrete Action Space
class AConvLSTMDiscrete(nn.Module): #DRQN
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, window_size=1, num_lstm_layers=1, action_bounds=1, dropout=0):
        super(AConvLSTMDiscrete, self).__init__()
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = int(hidden_size/2)
        self.action_bounds = action_bounds
        self.conv_layer = nn.Conv1d(observation_space, hidden_size, kernel_size=window_size)
        self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=self.lstm_hidden_size, num_layers=self.num_lstm_layers, batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(self.lstm_hidden_size, action_space)

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
        x = self.output_layer(x)
        return x, hx
    
    def act(self, state, hx=None) -> tuple[float, torch.Tensor, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs, hx = self.forward(state, hx)
        probs = F.softmax(probs, dim=1)       
        m = Categorical(probs) 
        action = m.sample()
        return (action.item() - 1), m.log_prob(action), hx


### ---------------- Continuous Action Space
class AConvLSTMContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, window_size=1, num_lstm_layers=1, action_bounds=1, dropout=0):
        super(AConvLSTMContinuous, self).__init__()
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
        return mean, std, hx
    
    def act(self, state, hx=None) -> tuple[float, torch.Tensor, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mean, std, hx = self.forward(state, hx)
        mean = torch.clamp(mean, -1, 1)
        std += 1e-5
        dist = Normal(mean, std) 
        action = dist.sample() 
        return torch.clamp(action, -1, 1).item(), dist.log_prob(action), hx












## ----------------- Convolutional
### ---------------- Discrete Action Space
class AConvDiscrete(nn.Module): #DQN
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, window_size=1, dropout=0.1):
        super(AConvDiscrete, self).__init__()
        self.conv_layer = nn.Conv1d(observation_space, hidden_size, kernel_size=window_size)
        self.output_layer = nn.Linear(hidden_size, action_space)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x) -> torch.Tensor:
        x = x.unsqueeze(-1).to(device)
        x = self.conv_layer(x)
        x = nn_activation_function(x)
        x = self.dropout_layer(x)
        x = x.squeeze().unsqueeze(0)
        x = self.output_layer(x)
        if len(x.shape) == 3: #Batch 
            x = x.squeeze() 
        return x

    def act(self, state) -> tuple[float, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu() 
        probs = F.softmax(probs, dim=1)       
        m = Categorical(probs) 
        action = m.sample() 
        return (action.item() - 1), m.log_prob(action)


### ---------------- Continuous Action Space
class AConvContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=1, window_size=1, action_bounds=1, dropout=0.1):
        super(AConvContinuous, self).__init__()
        self.action_bounds = action_bounds
        self.conv_layer = nn.Conv1d(observation_space, hidden_size, kernel_size=window_size)
        self.mean_layer = nn.Linear(hidden_size, 1)        
        self.std_layer = nn.Linear(hidden_size, 1)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x) -> torch.Tensor:
        x = x.unsqueeze(-1).to(device)
        x = self.conv_layer(x)
        x = nn_activation_function(x)
        x = self.dropout_layer(x)
        x = x.squeeze().unsqueeze(0)
        return torch.tanh(self.mean_layer(x)), F.softplus(self.std_layer(x))

    def act(self, state) -> tuple[float, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mean, std = self.forward(state)
        mean = torch.clamp(mean, -1, 1)
        std += 1e-5
        dist = Normal(mean, std) 
        action = dist.sample() 
        return torch.clamp(action, -1, 1).item(), dist.log_prob(action)



## ----------------- LSTM 
### ---------------- Discrete Action Space
"""
class ALSTMDiscrete(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, n_layers=1, dropout=0.1) -> None:
        super(ALSTMDiscrete, self).__init__()
        self.lstm_layer = nn.LSTM(input_size=observation_space, hidden_size=action_space, num_layers=n_layers, batch_first=True, dropout=dropout)
        #self.fcl2 = nn.Linear(hidden_size, action_space)        
        #self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, hx=None) -> tuple[torch.Tensor, torch.Tensor]:
        if hx is not None:
            x, hx = self.lstm_layer(x, hx)
        else: 
            x, hx = self.lstm_layer(x)
        #x = self.fcl2(x)
        return x, hx
    
    def act(self, state, hx=None) -> tuple[float, torch.Tensor, tuple]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs, hx = self.forward(state, hx)  
        probs = F.softmax(probs, dim=1)
        m = Categorical(probs) 
        action = m.sample() 
        return (action.item() - 1), m.log_prob(action), hx
"""
class ALSTMDiscrete(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, n_layers=1, dropout=0.1) -> None:
        super(ALSTMDiscrete, self).__init__()
        self.input_layer = nn.Linear(observation_space, hidden_size)        
        #self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=action_space, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_size, action_space)        
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, hx=None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_layer(x)
        x = nn_activation_function(x)
        x = self.dropout_layer(x)
        
        if hx is not None:
            x, hx = self.lstm_layer(x, hx)
        else: 
            x, hx = self.lstm_layer(x)
        x = nn_activation_function(x)
        x = self.output_layer(x)
        return x, hx
    
    def act(self, state, hx=None) -> tuple[float, torch.Tensor, tuple]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs, hx = self.forward(state, hx)  
        probs = F.softmax(probs, dim=1)
        m = Categorical(probs) 
        action = m.sample() 
        return (action.item() - 1), m.log_prob(action), hx
#"""

### ---------------- Continuous Action Space
class ALSTMContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, n_layers=1, dropout=0.1) -> None:
        super(ALSTMContinuous, self).__init__()
        self.input_layer = nn.Linear(observation_space, hidden_size)        
        self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.mean_layer = nn.Linear(hidden_size, 1)        
        self.std_layer = nn.Linear(hidden_size, 1)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, x, hx=None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_layer(x)
        x = nn_activation_function(x)
        x = self.dropout_layer(x)
        
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








## ----------------- FF 
### ---------------- Discrete Action Space
class FFDiscrete(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, dropout=0.1) -> None:
        super(FFDiscrete, self).__init__()
        self.fcl1 = nn.Linear(observation_space, hidden_size*2)
        self.fcl2 = nn.Linear(hidden_size*2, hidden_size)
        self.fcl3 = nn.Linear(hidden_size, action_space)
        self.dropout_layer = nn.Dropout(p=dropout)
        #self.bn1 = nn.BatchNorm1d(hidden_size*2) #
        #self.bn1 = nn.BatchNorm1d(1) #
        #self.bn2 = nn.BatchNorm1d(hidden_size) #

    def forward(self, x) -> torch.Tensor:
        x = self.fcl1(x)
        #x = self.bn1(x) #
        x = nn_activation_function(x)
        x = self.dropout_layer(x)
        x = self.fcl2(x)
        #x = self.bn2(x) #
        x = nn_activation_function(x)
        x = self.dropout_layer(x)
        x = self.fcl3(x)
        return x
    
    def act(self, state) -> tuple[float, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        probs = F.softmax(probs, dim=1)
        m = Categorical(probs) 
        action = m.sample() 
        return (action.item() - 1), m.log_prob(action)


### ---------------- Continuous Action Space
class AFFContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, dropout=0.1) -> None:
        super(AFFContinuous, self).__init__()        
        self.input_layer = nn.Linear(observation_space, hidden_size)        
        self.mean_layer = nn.Linear(hidden_size, 1)        
        self.std_layer = nn.Linear(hidden_size, 1)
        self.dropout_layer = nn.Dropout(p=dropout)
    
    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_layer(x)
        x = nn_activation_function(x)
        x = self.dropout_layer(x)
        mean = torch.tanh(self.mean_layer(x))
        std = F.softplus(self.std_layer(x))
        return mean, std
    
    def act(self, state) -> tuple[float, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mean, std = self.forward(state)
        mean = torch.clamp(mean, -1, 1)
        std += 1e-5
        dist = Normal(mean, std) 
        action = dist.sample() 
        return torch.clamp(action, -1, 1).item(), dist.log_prob(action)
