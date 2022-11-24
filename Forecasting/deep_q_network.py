import random
import numpy as np
from batch_learning import ReplayMemory, Transition, get_batch
import torch
torch.manual_seed(0)
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.MSELoss()


def act(model, state, hx=None, epsilon=0, recurrent=False) -> int:
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        if recurrent: 
            action_vals, hx = model(state, hx)
        else: 
            action_vals = model(state).cpu()
    if random.random() < epsilon:
        action = np.random.randint(-1, 2) #[-1, 2)
    else:
        action = action_vals.argmax().item() - 1
    return action, hx


def compute_loss_dqn(batch: tuple[torch.Tensor], net: torch.nn.Module, recurrent=False) -> torch.Tensor: 
    state_batch, action_batch, reward_batch, _ = batch
    reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + float(np.finfo(np.float32).eps))
    if recurrent:
        state_action_vals = net(state_batch)[0][range(action_batch.size(0)), (action_batch.long()+1)]
    else:
        state_action_vals = net(state_batch)[range(action_batch.size(0)), (action_batch.long()+1)] 
    return criterion(state_action_vals, reward_batch)


def update(replay_buffer: ReplayMemory, batch_size: int, net: torch.nn.Module, optimizer: torch.optim, recurrent=False) -> None:
    batch = get_batch(replay_buffer, batch_size)
    if batch is None:
        return
    optimizer.zero_grad()
    loss = compute_loss_dqn(batch, net, recurrent)
    loss.backward()
    #Pay attention to possible exploding gradient for certain hyperparameters
    #torch.nn.utils.clip_grad_norm_(net.parameters(), 1) 
    optimizer.step()


def deep_q_network(q_net, env, alpha=1e-4, weight_decay=1e-5, batch_size=10, exploration_rate=1, exploration_decay=(1-1e-3), exploration_min=0, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, train=True, print_res=True, print_freq=100, recurrent=False) -> tuple[np.ndarray, np.ndarray]: 
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
    optimizer = optim.Adam(q_net.parameters(), lr=alpha, weight_decay=weight_decay)
    replay_buffer = ReplayMemory(1000) # what capacity makes sense?
    reward_history = []
    action_history = []

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
            action, hx = act(q_net, state, hx, exploration_rate, recurrent) 
            next_state, reward, done, _ = env.step(action) 

            if done:
                break

            actions.append(action)
            rewards.append(reward)
            replay_buffer.push(torch.from_numpy(state).float().unsqueeze(0).to(device), 
                            torch.FloatTensor([action]), 
                            torch.FloatTensor([reward]), 
                            torch.from_numpy(next_state).float().unsqueeze(0).to(device))

            if train and len(replay_buffer) >= batch_size:
                update(replay_buffer, batch_size, q_net, optimizer, recurrent)

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
