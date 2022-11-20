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
nn_activation_function = nn.LeakyReLU()


class DQN(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, window_size=1) -> None:
        super(DQN, self).__init__() 
        self.conv_layer = nn.Conv1d(observation_space, hidden_size, kernel_size=window_size)
        self.output_layer = nn.Linear(hidden_size, action_space)

    def forward(self,x):
        x = x.unsqueeze(-1)
        x = self.conv_layer(x)
        x = nn_activation_function(x)
        x = x.squeeze().unsqueeze(0)
        x = self.output_layer(x)
        if len(x.shape) == 3: #Batch 
            x = x.squeeze()
        return x


def act(model, state, epsilon=0) -> int:
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    action_vals = model(state).cpu().detach()
    if random.random() < epsilon:
        action = np.random.choice(2) - 1
    else:
        action = action_vals.argmax().item() - 1
    return action


def compute_loss_dqn(batch: tuple[torch.Tensor], net: torch.nn.Module, target_net: torch.nn.Module) -> torch.Tensor: 
    state_batch, action_batch, reward_batch, next_state_batch = batch
    with torch.no_grad():
        target_state_vals = target_net(next_state_batch).max(1).values
    state_action_vals = net(state_batch)[range(action_batch.size(0)), (action_batch.long()+1)] 
    return criterion(state_action_vals, (reward_batch + target_state_vals))
    """
    loss = 0
    for s, a, r, s_ in zip(batch.state, batch.action, batch.reward, batch.next_state):
        loss += torch.nn.SmoothL1Loss().to(device)(net(s)[0, (a.long()+1)], (r + target_net(s_).max(1).values.squeeze())) / batch_size
    return loss
    #"""


def compute_loss_double_dqn(batch: tuple[torch.Tensor], net: torch.nn.Module, target_net: torch.nn.Module) -> torch.Tensor: 
    state_batch, action_batch, reward_batch, next_state_batch = batch
    with torch.no_grad():
        target_actions = net(next_state_batch).argmax(dim=1)
        target_state_vals = target_net(next_state_batch)[range(next_state_batch.size(0)), target_actions]
    state_action_vals = net(state_batch)[range(action_batch.size(0)), (action_batch.long()+1)] 
    return criterion(state_action_vals, (reward_batch + target_state_vals))


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


def deep_q_network(q_net, env, alpha=1e-5, weight_decay=1e-5, target_learning_rate=1e-1, batch_size=10, exploration_rate=0.1, exploration_decay=(1-1e-2), exploration_min=0, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, double_dqn = False, train=True, print_res=True, print_freq=100) -> tuple[np.ndarray, np.ndarray]: 
    """
    Training for DQN

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
    replay_buffer = ReplayMemory(1000) # what capacity makes sense?
    reward_history = []
    action_history = []

    if double_dqn:
        loss_fn = compute_loss_double_dqn
    else: 
        loss_fn = compute_loss_dqn
    if not train:
        exploration_rate = exploration_min
    
    for n in range(num_episodes):
        rewards = []
        actions = []
        state = env.reset() 

        for i in range(max_episode_length): 
            action = act(q_net, state, exploration_rate) 
            next_state, reward, done, _ = env.step(action) 

            if done:
                if train:
                    update(replay_buffer, max(2, (i%batch_size)), q_net, target_net, optimizer, loss_fn)
                break

            actions.append(action)
            rewards.append(reward)
            replay_buffer.push(torch.from_numpy(state).float().unsqueeze(0).to(device), 
                            torch.FloatTensor([action]), 
                            torch.FloatTensor([reward]), 
                            torch.from_numpy(next_state).float().unsqueeze(0).to(device))

            if train and i % batch_size == 0:
                update(replay_buffer, batch_size, q_net, target_net, optimizer, loss_fn)
                soft_updates(q_net, target_net, target_learning_rate)
            
            state = next_state
            exploration_rate = max(exploration_rate*exploration_decay, exploration_min)
        
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
