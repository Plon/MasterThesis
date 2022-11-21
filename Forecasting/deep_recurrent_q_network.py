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
from deep_q_network import update
#criterion = torch.nn.SmoothL1Loss()
criterion = torch.nn.MSELoss()


#TODO improve random exploration
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
    reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + float(np.finfo(np.float32).eps))
    with torch.no_grad():
        max_q_next = target_net(next_state_batch)[0].max(1).values
    q_est = max_q_next + reward_batch
    q = net(state_batch)[0][range(action_batch.size(0)), (action_batch.long()+1)]
    return criterion(q, q_est)


def compute_loss_double_dqn(batch: tuple[torch.Tensor], net: torch.nn.Module, target_net: torch.nn.Module) -> torch.Tensor: 
    state_batch, action_batch, reward_batch, next_state_batch = batch
    reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + float(np.finfo(np.float32).eps))
    with torch.no_grad():
        target_actions = net(next_state_batch)[0].argmax(dim=1)
        max_q_next = target_net(next_state_batch)[0][range(next_state_batch.size(0)), target_actions]
    q_est = max_q_next + reward_batch
    q = net(state_batch)[0][range(action_batch.size(0)), (action_batch.long()+1)]
    return criterion(q, q_est)


def deep_recurrent_q_network(q_net, env, alpha=1e-4, weight_decay=1e-5, target_learning_rate=1e-1, batch_size=10, exploration_rate=0.1, exploration_decay=(1-1e-2), exploration_min=0, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, double_dqn = False, train=True, print_res=True, print_freq=100) -> tuple[np.ndarray, np.ndarray]: 
    """
    Training for DRQN

    Args: 
        q_net (): network that parameterizes the Q-learning function
        env: environment that the rl agent interacts with
        alpha (float): learning rate
        weight_decay (float): regularization 
        target_learning_rate (int): 
        batch_size (int): 
        exploration_rate (float): 
        exploration_decay (float): 
        exploration_min (float): 
        num_episodes (int): 
        max_episode_length (int): 
        double_dqn (bool): 
        train (bool): training mode if true
        print_res (bool): printing results if true
        print_freq (int): at what episode interval to print results
    Returns: 
        reward_history (numpy.ndarray): the rewards of each episode
        action_history (numpy.ndarray): the actions chosen by the agent        
    """
    target_net = deepcopy(q_net)
    target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=alpha, weight_decay=weight_decay)
    replay_buffer = ReplayMemory(1000) # what capacity makes sense?
    reward_history = []
    action_history = []

    if double_dqn:
        loss_fn = compute_loss_double_dqn
    else: 
        loss_fn = compute_loss_dqn
    
    if not train:
        exploration_min = 0
        exploration_rate = exploration_min
        q_net.eval()
    else:
        q_net.train()
    
    for n in range(num_episodes):
        rewards = []
        actions = []
        state = env.reset() 
        hx = None   

        for i in range(max_episode_length):
            action, hx = act(q_net, state, hx, exploration_rate) 
            next_state, reward, done, _ = env.step(action) 

            if done:
                if train:
                    update(replay_buffer, min((i+1), batch_size), q_net, target_net, optimizer, loss_fn)
                break

            actions.append(action)
            rewards.append(reward)
            replay_buffer.push(torch.from_numpy(state).float().unsqueeze(0).to(device), 
                            torch.FloatTensor([action]), 
                            torch.FloatTensor([reward]), 
                            torch.from_numpy(next_state).float().unsqueeze(0).to(device))

            if train and i >= batch_size:
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
