import random
from copy import deepcopy
import numpy as np
from batch_learning import ReplayMemory, Transition, get_batch
import torch
torch.manual_seed(0)
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from deep_deterministic_policy_gradient import soft_updates
#criterion = torch.nn.SmoothL1Loss()
criterion = torch.nn.MSELoss()


class DRQN(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, window_size=1, num_lstm_layers=1) -> None:
        super(DRQN, self).__init__()
        self.input_layer = nn.Conv1d(observation_space, hidden_size, kernel_size=window_size)
        self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=action_space, num_layers=num_lstm_layers, batch_first=True)

    def forward(self, x, hx=None) -> tuple[torch.Tensor, torch.Tensor]:
        #Conv layer
        x = x.unsqueeze(-1) 
        x = self.input_layer(x)
        x = F.relu(x)
        x = x.squeeze().unsqueeze(0)
        if len(x.shape) == 3: #Batch 
            x = x.squeeze()

        #LSTM layer
        if hx is not None:
            x, hx = self.lstm_layer(x, hx)
        else: 
            x, hx = self.lstm_layer(x)
        return x, hx


class DLSTMQN(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, num_lstm_layers=1) -> None:
        super(DLSTMQN, self).__init__()
        self.input_layer = nn.Linear(observation_space, hidden_size)
        self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=action_space, num_layers=num_lstm_layers, batch_first=True)

    def forward(self, x, hx=None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_layer(x)
        x = F.relu(x)        
        if hx is not None:
            x, hx = self.lstm_layer(x, hx)
        else: 
            x, hx = self.lstm_layer(x)
        return x, hx


def act(model, state, hx, epsilon=0) -> tuple[int, torch.Tensor]:
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    action_vals, hx = model(state, hx)
    if random.random() < epsilon:
        action = np.random.choice(2) - 1
    else:
        action = action_vals.argmax().item() - 1
    return action, hx


def compute_loss_dqn(batch: tuple[torch.Tensor], net: torch.nn.Module, target_net: torch.nn.Module) -> torch.Tensor: 
    state_batch, action_batch, reward_batch, next_state_batch = batch
    with torch.no_grad():
        max_q_next = target_net(next_state_batch)[0].max(1).values
    q_est = max_q_next + reward_batch
    q = net(state_batch)[0][range(action_batch.size(0)), (action_batch.long()+1)]
    return criterion(q, q_est)


def compute_loss_double_dqn(batch: tuple[torch.Tensor], net: torch.nn.Module, target_net: torch.nn.Module) -> torch.Tensor: 
    state_batch, action_batch, reward_batch, next_state_batch = batch
    with torch.no_grad():
        target_actions = net(next_state_batch)[0].argmax(dim=1)
        max_q_next = target_net(next_state_batch)[0][range(next_state_batch.size(0)), target_actions]
    q_est = max_q_next + reward_batch
    q = net(state_batch)[0][range(action_batch.size(0)), (action_batch.long()+1)]
    return criterion(q, q_est)


def update(replay_buffer: ReplayMemory, batch_size: int, net: torch.nn.Module, target_net: torch.nn.Module, optimizer: torch.optim, loss_fn=compute_loss_dqn) -> None:
    batch = get_batch(replay_buffer, batch_size)
    if batch is None:
        return
    optimizer.zero_grad()
    loss = loss_fn(batch, net, target_net)
    loss.backward()
    #Pay attention to possible exploding gradient for certain hyperparameters
    #torch.nn.utils.clip_grad_norm_(net.parameters(), 1) 
    optimizer.step()


def deep_recurrent_q_network(q_net, env, alpha=1e-5, weight_decay=1e-5, target_learning_rate=1e-1, batch_size=10, exploration_rate=0.1, exploration_decay=(1-1e-2), exploration_min=0.005, num_episodes=np.iinfo(np.int32).max, double_dqn = False) -> tuple[np.ndarray, np.ndarray]: 
    """
    Training for DRQN

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
    hx = None
    if double_dqn:
        loss_fn = compute_loss_double_dqn
    else: 
        loss_fn = compute_loss_dqn
        
    for i in range(num_episodes):
        action, hx = act(q_net, state, hx, exploration_rate) 
        next_state, reward, done, _ = env.step(action) 

        if done:
            update(replay_buffer, max(2, (i%batch_size)), q_net, target_net, optimizer, loss_fn)
            break

        actions.append(action)
        rewards.append(reward)
        replay_buffer.push(torch.from_numpy(state).float().unsqueeze(0).to(device), 
                           torch.FloatTensor([action]), 
                           torch.FloatTensor([reward]), 
                           torch.from_numpy(next_state).float().unsqueeze(0).to(device))

        if i % batch_size == 0:
            update(replay_buffer, batch_size, q_net, target_net, optimizer, loss_fn)
            soft_updates(q_net, target_net, target_learning_rate)

        state = next_state
        exploration_rate = max(exploration_rate*exploration_decay, exploration_min)

    return np.array(rewards), np.array(actions)
