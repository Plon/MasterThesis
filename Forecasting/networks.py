import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#nn_activation_function = nn.ReLU()
nn_activation_function = nn.LeakyReLU()
#nn_activation_function = nn.ELU()


def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

def _get_seq_out(seq, shape) -> int:
    o = seq(torch.zeros(1, 1, shape))
    return int(np.prod(o.size())) 

## ----------------- Convolutional + LSTM
### ---------------- Discrete Action Space
class AConvLSTMDiscrete(nn.Module): #DRQN
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, window_size=1, num_lstm_layers=1, action_bounds=1, dropout=0):
        super(AConvLSTMDiscrete, self).__init__()  
        self.fc_in = nn.Sequential(
            nn.Linear(observation_space, hidden_size), 
            nn_activation_function,
            nn.Dropout(p=dropout), 
        )
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=4, stride=2), 
            nn_activation_function, 
            nn.BatchNorm1d(hidden_size), #seems like batch norm functions better after activation than before
            nn.Dropout(p=dropout), 
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1), 
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn_activation_function,
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=dropout), 
        )
        conv_out = _get_seq_out(self.conv, hidden_size)
        self.lstm_layer = nn.LSTM(input_size=conv_out, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, action_space)

    def forward(self, x, hx=None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.fc_in(x)
        x = x.unsqueeze(1) 
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        if hx is not None:
            x, hx = self.lstm_layer(x, hx)
        else: 
            x, hx = self.lstm_layer(x)
        x = nn_activation_function(x)
        x = self.fc_out(x)
        return x, hx


### ---------------- Continuous Action Space
class AConvLSTMContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, num_lstm_layers=1, dropout=0):
        super(AConvLSTMContinuous, self).__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(observation_space, hidden_size), 
            nn_activation_function,
            nn.Dropout(p=dropout), 
        )
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=4, stride=2), 
            nn_activation_function, 
            nn.BatchNorm1d(hidden_size), #seems like batch norm functions better after activation than before
            nn.Dropout(p=dropout), 
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1), 
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn_activation_function,
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=dropout), 
        )
        conv_out = _get_seq_out(self.conv, hidden_size)
        self.lstm_layer = nn.LSTM(input_size=conv_out, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=True, dropout=dropout)
        self.mean_layer = nn.Linear(hidden_size, 1)        
        self.std_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, hx=None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.fc_in(x)
        x = x.unsqueeze(1) 
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        if hx is not None:
            x, hx = self.lstm_layer(x, hx)
        else: 
            x, hx = self.lstm_layer(x)
        x = nn_activation_function(x)
        mean = torch.tanh(self.mean_layer(x))
        std = F.softplus(self.std_layer(x))
        return mean, std, hx


## ----------------- Convolutional
### ---------------- Discrete Action Space
class AConvDiscrete(nn.Module): #DQN
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, dropout=0.1):
        super(AConvDiscrete, self).__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(observation_space, hidden_size), 
            nn_activation_function,
            nn.Dropout(p=dropout), 
        )
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=4, stride=2), 
            nn_activation_function, 
            nn.BatchNorm1d(hidden_size), #seems like batch norm functions better after activation than before
            nn.Dropout(p=dropout), 
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1), 
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn_activation_function,
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=dropout), 
        )
        conv_out = _get_seq_out(self.conv, hidden_size)
        self.fc_out = nn.Linear(conv_out, action_space)

    def forward(self, x) -> torch.Tensor:
        x = self.fc_in(x)
        x = x.unsqueeze(1) #Think you have to do this since there is no batch_first arg for conv nets
        x = self.conv(x)
        x = x.view(x.shape[0], -1) #Add all activation maps to one big activation map
        x = self.fc_out(x)
        return x


### ---------------- Continuous Action Space
class AConvContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, dropout=0.1):
        super(AConvContinuous, self).__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(observation_space, hidden_size), 
            nn_activation_function,
            nn.Dropout(p=dropout), 
        )
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=4, stride=2), 
            nn_activation_function, 
            nn.BatchNorm1d(hidden_size), 
            nn.Dropout(p=dropout), 
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1), 
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn_activation_function,
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=dropout), 
        )
        conv_out = _get_seq_out(self.conv, hidden_size)
        self.mean_layer = nn.Linear(conv_out, 1)        
        self.std_layer = nn.Linear(conv_out, 1)       

    def forward(self, x) -> torch.Tensor:
        x = self.fc_in(x)
        x = x.unsqueeze(1) 
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        mean = torch.tanh(self.mean_layer(x))
        std = F.softplus(self.std_layer(x))
        return mean, std


## ----------------- LSTM 
### ---------------- Discrete Action Space
# Seems like performance increases if there is an activation function after lstm layer
class ALSTMDiscrete(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, n_layers=1, dropout=0.1) -> None:
        super(ALSTMDiscrete, self).__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(observation_space, hidden_size), 
            nn_activation_function,
            nn.Dropout(p=dropout), 
        )
        self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, action_space)        

    def forward(self, x, hx=None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.fc_in(x)
        if hx is not None:
            x, hx = self.lstm_layer(x, hx)
        else: 
            x, hx = self.lstm_layer(x)
        x = nn_activation_function(x)
        x = self.fc_out(x)
        return x, hx


### ---------------- Continuous Action Space
class ALSTMContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, n_layers=1, dropout=0.1) -> None:
        super(ALSTMContinuous, self).__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(observation_space, hidden_size), 
            nn_activation_function,
            nn.Dropout(p=dropout), 
        )  
        self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.mean_layer = nn.Linear(hidden_size, 1)        
        self.std_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, hx=None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.fc_in(x)
        if hx is not None:
            x, hx = self.lstm_layer(x, hx)
        else: 
            x, hx = self.lstm_layer(x)
        x = nn_activation_function(x)
        mean = torch.tanh(self.mean_layer(x))
        std = F.softplus(self.std_layer(x))
        return mean, std, hx


## ----------------- FF 
### ---------------- Discrete Action Space
class FFDiscrete(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, dropout=0.1) -> None:
        super(FFDiscrete, self).__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(observation_space, hidden_size*2), 
            nn_activation_function,
            nn.Dropout(p=dropout), 
            nn.Linear(hidden_size*2, hidden_size), 
            nn_activation_function,
            nn.Dropout(p=dropout), 
        )  
        self.fc_out = nn.Linear(hidden_size, action_space)

    def forward(self, x) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.fc_out(x)
        return x


### ---------------- Continuous Action Space
class AFFContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, dropout=0.1) -> None:
        super(AFFContinuous, self).__init__()    
        self.fc_in = nn.Sequential(
            nn.Linear(observation_space, hidden_size*2), 
            nn_activation_function,
            nn.Dropout(p=dropout), 
            nn.Linear(hidden_size*2, hidden_size), 
            nn_activation_function,
            nn.Dropout(p=dropout), 
        )  
        self.mean_layer = nn.Linear(hidden_size, 1)        
        self.std_layer = nn.Linear(hidden_size, 1)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.fc_in(x)
        mean = torch.tanh(self.mean_layer(x))
        std = F.softplus(self.std_layer(x))
        return mean, std


## ----------------- Linear 
### ---------------- Discrete Action Space
class LinearDiscrete(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, dropout=0.1) -> None:
        super(LinearDiscrete, self).__init__()
        self.fc_out = nn.Linear(observation_space, action_space)

    def forward(self, x) -> torch.Tensor:
        x = self.fc_out(x)
        return x


### ---------------- Continuous Action Space
class LinearContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=1, dropout=0.1) -> None:
        super(LinearContinuous, self).__init__()    
        self.fc_out = nn.Linear(observation_space, action_space)
        self.std = torch.tensor(np.ones((action_space, )) * 0.1).squeeze() 

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.fc_out(x) 
        return mean, self.std


# ------------------ State-Action
## ----------------- Conv
class CConvSA(nn.Module):
    def __init__(self, observation_space=8, hidden_size=64, action_space=1, dropout=0.1) -> None:
        super(CConvSA, self).__init__()         
        self.fc1 = nn.Linear(observation_space,hidden_size*2)
        self.fc2 = nn.Linear(((hidden_size*2)+action_space),hidden_size)
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=4, stride=2), 
            nn_activation_function, 
            nn.BatchNorm1d(hidden_size), 
            nn.Dropout(p=dropout), 
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1), 
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn_activation_function,
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p=dropout), 
        )
        conv_out = _get_seq_out(self.conv, hidden_size)
        self.fc3 = nn.Linear(conv_out,1)
        
    def forward(self, state, action):
        x = self.fc1(state)
        x = nn_activation_function(x)
        x = torch.cat((x, action), dim=1)
        x = self.fc2(x)
        x = nn_activation_function(x)
        x = x.unsqueeze(1) 
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc3(x)
        return x


## ----------------- FF
class CFFSA(nn.Module):
    def __init__(self, observation_space=8, hidden_size=64, action_space=1, dropout=0.1) -> None:
        super(CFFSA, self).__init__()         
        self.fc1 = nn.Linear(observation_space,hidden_size*2)
        self.fc2 = nn.Linear(((hidden_size*2)+action_space),hidden_size)
        self.fc3 = nn.Linear(hidden_size,1)
        
    def forward(self, state, action):
        x = self.fc1(state)
        x = nn_activation_function(x)
        x = torch.cat((x, action), dim=1)
        x = self.fc2(x)
        x = nn_activation_function(x)
        x = self.fc3(x)
        return x
