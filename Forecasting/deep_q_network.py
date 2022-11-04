import random
from copy import deepcopy
import numpy as np
from batch_learning import ReplayMemory, Transition
import torch
torch.manual_seed(0)
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from deep_deterministic_policy_gradient import soft_updates


#TODO fix intelligent exploring
def act(model, state, epsilon=0) -> int:
    if random.random() < epsilon:
        return np.random.choice(2) - 1
    else:
        #return model(torch.from_numpy(state).float().unsqueeze(0).to(device)).cpu().argmax().item() - 1
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            action_vals = model(state).cpu()
        model.train()
        print(action_vals, action_vals.argmax().item())
        return action_vals.argmax().item() - 1


class DQN(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, window_size=1) -> None:
        super(DQN, self).__init__() 
        self.input_layer = nn.Conv1d(observation_space, hidden_size, kernel_size=window_size)
        self.output_layer = nn.Linear(hidden_size, action_space)

    def forward(self,x):
        x = x.unsqueeze(-1)
        x = F.relu(self.input_layer(x))
        x = x.squeeze().unsqueeze(0)
        x = self.output_layer(x)
        if len(x.shape) == 3: #Batch 
            x = x.squeeze()
        return x


class DRQN(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, window_size=1, num_lstm_layers=1) -> None:
        super(DRQN, self).__init__()
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = int(hidden_size/2)
        self.input_layer = nn.Conv1d(observation_space, hidden_size, kernel_size=window_size)
        self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=self.lstm_hidden_size, num_layers=self.num_lstm_layers, batch_first=True)
        self.output_layer = nn.Linear(self.lstm_hidden_size, action_space)

    def forward(self, x) -> torch.Tensor:
        #Conv layer - can be removed
        x = x.unsqueeze(-1) 
        x = F.relu(self.input_layer(x))
        x = x.squeeze().unsqueeze(0)
        if len(x.shape) == 3: #Batch 
            x = x.squeeze()

        #LSTM layer
        h_0 = Variable(torch.zeros(self.num_lstm_layers, self.lstm_hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_lstm_layers, self.lstm_hidden_size)) #internal state
        x, (hx, cx) = self.lstm_layer(x, (h_0, c_0))
        x = self.output_layer(F.relu(x))
        return x


#TODO seems to favour one action after a while
def batch_learning(replay_buffer: ReplayMemory, batch_size: int, q_net: torch.nn.Module, target_net: torch.nn.Module, optimizer: torch.optim) -> None: 
    """
    Performs batch learning on the q_net

    Args: 
        replay_buffer (ReplayMemory): memory that (S, A, R, S') experience is drawn from
        batch_size (int): number of batches to train the network on
        q_net (nn.Module): the network to be trained
        target_net (nn.Module): the target network to q_net
        optimizer (nn.Optim): the optimizer for q_net
    """
    if len(replay_buffer) < batch_size:
        return
    if batch_size < 2:
        raise ValueError("Argument batch_size must be integer >= 2")
    
    batch = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*batch))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action) 
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    with torch.no_grad():
        target_state_vals = target_net(next_state_batch).max(1).values.detach()    
    state_action_vals = q_net(state_batch)[range(action_batch.size(0)), (action_batch.long()+1)] 
    loss = torch.nn.SmoothL1Loss()(state_action_vals, (reward_batch + target_state_vals))
    """
    loss = 0
    for s, a, r, s_ in zip(batch.state, batch.action, batch.reward, batch.next_state):
        loss += torch.nn.SmoothL1Loss().to(device)(q_net(s)[0, (a.long()+1)], (r + target_net(s_).max(1).values.squeeze())) / batch_size
    #"""
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1) #Exploding gradient 
    optimizer.step()


def deep_q_network(q_net, env, alpha=1e-5, weight_decay=1e-5, target_learning_rate=1e-1, batch_size=10, exploration_rate=0.1, exploration_decay=(1-1e-2), exploration_min=0.005, num_episodes=np.iinfo(np.int32).max) -> tuple[np.ndarray, np.ndarray]: 
    """
    Training for DQN or DRQN

    Args: 
        q_net (): network that parameterizes the Q-learning function
        env: environment that the rl agent interacts with
        leraning_rate (float): 
        weight_decay (float): regularization 
        target_update_freq (int): 
        batch_size (int): 
        exploration_rate (float): 
        exploration_decay (float): 
        exploration_min (float): 
        num_episodes (int): maximum number of episodes
    Returns: 
        scores (numpy.ndarray): the rewards of each episode
        actions (numpy.ndarray): the actions chosen by the agent        
    """
    target_net = deepcopy(q_net)
    optimizer = optim.Adam(q_net.parameters(), lr=alpha, weight_decay=weight_decay)
    rewards = []
    actions = []
    replay_buffer = ReplayMemory(1000) # what capacity makes sense?
    state = env.state() 

    for i in range(num_episodes):
        action = act(q_net, state, exploration_rate) 
        next_state, reward, done, _ = env.step(action) 

        if done:
            batch_learning(replay_buffer, max(2, (i % batch_size)), q_net, target_net, optimizer)
            break

        actions.append(action)
        rewards.append(reward)
        replay_buffer.push(torch.from_numpy(state).float().unsqueeze(0).to(device), torch.FloatTensor([action]), torch.FloatTensor([reward]), torch.from_numpy(next_state).float().unsqueeze(0).to(device))

        if i % batch_size == 0:
            batch_learning(replay_buffer, batch_size, q_net, target_net, optimizer)
            soft_updates(q_net, target_net, target_learning_rate)
        
        state = next_state
        exploration_rate = max(exploration_rate*exploration_decay, exploration_min)

    return np.array(rewards), np.array(actions)
