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


def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)


# ------------------ Actor Networks (Policy)
## ----------------- Convolutional + LSTM
### ---------------- Discrete Action Space
class ActorNetwork1DConvolutionalLSTMDiscrete(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, window_size=1, num_lstm_layers=1):
        super(ActorNetwork1DConvolutionalLSTMDiscrete, self).__init__()
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = int(hidden_size/2)
        self.input_layer = nn.Conv1d(observation_space, hidden_size, kernel_size=window_size)
        self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=self.lstm_hidden_size, num_layers=self.num_lstm_layers, batch_first=True)
        self.output_layer = nn.Linear(self.lstm_hidden_size, action_space)

    def forward(self, x) -> torch.Tensor:
        #Conv layer
        x = x.unsqueeze(-1).to(device)
        x = self.input_layer(x)
        x = F.relu(x)
        x = x.squeeze().unsqueeze(0)

        #LSTM layer
        h_0 = Variable(torch.zeros(self.num_lstm_layers, self.lstm_hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_lstm_layers, self.lstm_hidden_size)) #internal state
        x, (hx, cx) = self.lstm_layer(x, (h_0, c_0))
        x = self.output_layer(F.relu(x))
        return F.softmax(x, dim=1)

    def act(self, state) -> tuple[float, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()        
        m = Categorical(probs) 
        action = m.sample() 
        return (action.item() - 1), m.log_prob(action)


### ---------------- Continuous Action Space
class ActorNetwork1DConvolutionalLSTMContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=1, window_size=1, num_lstm_layers=1, action_bounds=1):
        super(ActorNetwork1DConvolutionalLSTMContinuous, self).__init__()
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = int(hidden_size/2)
        self.action_bounds = action_bounds
        self.input_layer = nn.Conv1d(observation_space, hidden_size, kernel_size=window_size)
        self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=self.lstm_hidden_size, num_layers=self.num_lstm_layers, batch_first=True)
        self.mean_layer = nn.Linear(self.lstm_hidden_size, 1)        
        self.std_layer = nn.Linear(self.lstm_hidden_size, 1)

    def forward(self, x) -> torch.Tensor:
        #Conv layer
        x = x.unsqueeze(-1).to(device)
        x = self.input_layer(x)
        x = F.relu(x)
        x = x.squeeze().unsqueeze(0)

        #LSTM layer
        h_0 = Variable(torch.zeros(self.num_lstm_layers, self.lstm_hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_lstm_layers, self.lstm_hidden_size)) #internal state
        x, (hx, cx) = self.lstm_layer(x, (h_0, c_0))
        x = F.relu(x)
                
        #Output layer
        return torch.tanh(self.mean_layer(x)), F.softplus(self.std_layer(x))

    def act(self, state) -> tuple[float, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mean, std = self.forward(state)
        mean = torch.clamp(mean, -1, 1)
        std += 1e-5
        dist = Normal(mean, std) 
        action = dist.sample() 
        return torch.clamp(action, -1, 1).item(), dist.log_prob(action)


## ----------------- Convolutional
### ---------------- Discrete Action Space
class ActorNetwork1DConvolutionalDiscrete(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, window_size=1):
        super(ActorNetwork1DConvolutionalDiscrete, self).__init__()
        self.input_layer = nn.Conv1d(observation_space, hidden_size, kernel_size=window_size)
        self.output_layer = nn.Linear(hidden_size, action_space)

    def forward(self, x) -> torch.Tensor:
        x = x.unsqueeze(-1).to(device)
        x = self.input_layer(x)
        x = F.relu(x)
        x = x.squeeze().unsqueeze(0)
        x = self.output_layer(x)
        return F.softmax(x, dim=1)

    def act(self, state) -> tuple[float, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()        
        m = Categorical(probs) 
        action = m.sample() 
        return (action.item() - 1), m.log_prob(action)


### ---------------- Continuous Action Space
class ActorNetwork1DConvolutionalContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=1, window_size=1, action_bounds=1):
        super(ActorNetwork1DConvolutionalContinuous, self).__init__()
        self.action_bounds = action_bounds
        self.input_layer = nn.Conv1d(observation_space, hidden_size, kernel_size=window_size)
        self.mean_layer = nn.Linear(hidden_size, 1)        
        self.std_layer = nn.Linear(hidden_size, 1)

    def forward(self, x) -> torch.Tensor:
        x = x.unsqueeze(-1).to(device)
        x = self.input_layer(x)
        x = F.relu(x)
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
class ActorNetworkLSTMDiscrete(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, n_layers=1) -> None:
        super(ActorNetworkLSTMDiscrete, self).__init__()
        self.input_layer = nn.Linear(observation_space, hidden_size)
        #self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=action_space, num_layers=n_layers, batch_first=True)
        self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, action_space)

    def forward(self, x) -> torch.Tensor:
        x = self.input_layer(x)
        x = F.relu(x)   
        x, (hx, cx) = self.lstm_layer(x)
        #print(x.shape)
        x = F.relu(hx)
        x = self.output_layer(x)
        #print(x.shape)
        return F.softmax(x, dim=1)

    def act(self, state) -> tuple[int, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()        
        m = Categorical(probs) 
        action = m.sample() 
        return (action.item() - 1), m.log_prob(action)


### ---------------- Continuous Action Space
class ActorNetworkLSTMContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=1, n_layers=1, action_bounds=1) -> None:
        super(ActorNetworkLSTMContinuous, self).__init__()
        self.action_bounds = action_bounds
        self.num_layers = n_layers
        self.hidden_size = hidden_size
        self.lstm_layer = nn.LSTM(input_size=observation_space, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.mean_layer = nn.Linear(hidden_size, 1)        
        self.std_layer = nn.Linear(hidden_size, 1)

    def forward(self, x) -> torch.Tensor:
        #h_0 = Variable(torch.zeros(self.num_layers, self.hidden_size)) #hidden state
        #c_0 = Variable(torch.zeros(self.num_layers, self.hidden_size)) #internal state
        #x, (hx, cx) = self.lstm_layer(x, (h_0, c_0))
        x, (hx, cx) = self.lstm_layer(x)
        #x = F.relu(x)
        print(x.shape, hx.shape)
        return torch.tanh(self.mean_layer(x)), F.softplus(self.std_layer(x))
    
    def act(self, state) -> tuple[float, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mean, std = self.forward(state)
        mean = torch.clamp(mean, -1, 1)
        std += 1e-5
        dist = Normal(mean, std) 
        action = dist.sample() 
        return torch.clamp(action, -1, 1).item(), dist.log_prob(action)


## ----------------- FF 
### ---------------- Discrete Action Space
class ActorNetworkDiscrete(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3) -> None:
        super(ActorNetworkDiscrete, self).__init__()
        self.input_layer = nn.Linear(observation_space, hidden_size)
        self.output_layer = nn.Linear(hidden_size, action_space)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.input_layer(x))
        x = self.output_layer(x)
        return F.softmax(x, dim=1)
    
    def act(self, state) -> tuple[float, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()        
        m = Categorical(probs) 
        action = m.sample() 
        return (action.item() - 1), m.log_prob(action)


### ---------------- Continuous Action Space
class ActorNetworkContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128) -> None:
        super(ActorNetworkContinuous, self).__init__()        
        self.input_layer = nn.Linear(observation_space, hidden_size)        
        self.mean_layer = nn.Linear(hidden_size, 1)        
        self.std_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.input_layer(x))
        return torch.tanh(self.mean_layer(x)), F.softplus(self.std_layer(x))
    
    def act(self, state) -> tuple[float, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mean, std = self.forward(state)
        mean = torch.clamp(mean, -1, 1)
        std += 1e-5
        dist = Normal(mean, std) 
        action = dist.sample() 
        return torch.clamp(action, -1, 1).item(), dist.log_prob(action)


# ------------------ Critic Networks (Value)
## ----------------- FF
class CriticNetwork(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=1) -> None:
        super(CriticNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, hidden_size)
        self.output_layer = nn.Linear(hidden_size, action_space)

    def forward(self, x) -> torch.Tensor:
        x = self.input_layer(x)
        x = F.relu(x)
        return self.output_layer(x)
