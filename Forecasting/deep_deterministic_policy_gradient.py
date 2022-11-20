from copy import deepcopy
import numpy as np
import torch
torch.manual_seed(0)
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from batch_learning import ReplayMemory, Transition, get_batch
from reinforce import optimize
nn_activation_function = nn.LeakyReLU()


class DDPG_Actor(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=1, EPS=0.003) -> None:
        super(DDPG_Actor, self).__init__()
        self.fc1 = nn.Linear(observation_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_space)

    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = nn_activation_function(x)
        x = self.fc2(x)        
        x = torch.tanh(x)
        return x


class DDPG_Critic(nn.Module):
    def __init__(self, observation_space=8, hidden_size=64, action_space=1, EPS=0.003) -> None:
        super(DDPG_Critic, self).__init__()         
        self.fc1 = nn.Linear(observation_space,hidden_size)
        self.fc2 = nn.Linear(hidden_size+action_space,int(hidden_size/2))
        self.fc3 = nn.Linear(int(hidden_size/2),1)
        
    def forward(self, state, action):
        x = self.fc1(state)
        x = nn_activation_function(x)
        x = torch.cat((x, action), dim=1)
        x = self.fc2(x)
        x = nn_activation_function(x)
        x = self.fc3(x)
        return x


def act(net, state, epsilon=0, training=True) -> torch.Tensor:
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        action = net(state).cpu()#.detach()
    noise = ((np.random.rand(1)[0] * 2) - 1) #TODO maybe change to Ornstein-Uhlenbeck process
    action += training*max(epsilon, 0)*noise
    action = torch.clamp(action, -1, 1).item()
    return action


def soft_updates(net, target_net, tau):
    """ Ø' <- tØ' + (1-t)Ø """
    if tau > 1 or tau < 0:
        raise ValueError("Argument tau must be number on the interval [0,1]")
    with torch.no_grad():
        for p, p_targ in zip(net.parameters(), target_net.parameters()):
            p_targ.data.mul_(tau)
            p_targ.data.add_((1 - tau) * p.data)


def update(replay_buffer: ReplayMemory, batch_size: int, critic: torch.nn.Module, critic_target: torch.nn.Module, actor: torch.nn.Module, actor_target: torch.nn.Module, optimizer_critic: torch.optim, optimizer_actor: torch.optim) -> None: 
    # Retrieve batch
    batch = get_batch(replay_buffer, batch_size)
    if batch is None:
        return
    
    #Compute loss and optimize critic
    c_loss = compute_critic_loss(actor_target, critic, critic_target, batch)
    optimize(optimizer_critic, c_loss)

    # Freeze Q-net
    for p in critic.parameters(): 
        p.requires_grad = False

    # Compute loss and optimize actor 
    a_loss = compute_actor_loss(actor, critic, batch[0])
    optimize(optimizer_actor, a_loss)

    # Unfreeze Q-net
    for p in critic.parameters(): 
        p.requires_grad = True


def compute_actor_loss(mu, q, s) -> torch.Tensor: 
    """ Returns policy loss -Q(s, mu(s)) """
    mu_s = mu(s)
    q_sa = q(s, mu_s)
    loss = -1*torch.mean(q_sa)
    return loss


def compute_critic_loss(actor_target, critic, critic_target, batch) -> torch.Tensor: 
    """ Returns Q loss Q(s_t, a) - (R_t+1 + Q'(s_t+1, mu'(s_t+1))) """
    state, action, reward, next_state = batch
    with torch.no_grad():
        a_hat = actor_target(next_state)
        q_sa_hat = critic_target(next_state, a_hat).squeeze()
    q_sa = critic(state, action.reshape(-1, 1)).squeeze()
    loss = torch.nn.MSELoss()(q_sa, (reward + q_sa_hat))
    return loss


def deep_determinstic_policy_gradient(actor_net, critic_net, env, alpha_actor=1e-3, alpha_critic=1e-3, weight_decay=1e-3, target_learning_rate=1e-1, batch_size=30, exploration_rate=0.1, exploration_decay=(1-1e-2), exploration_min=0, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, train=True, print_res=True, print_freq=100) -> tuple[np.ndarray, np.ndarray]: 
    """
    Training for DDPG

    Args: 
        actor_net (): network that parameterizes the policy
        critic_net (): network that parameterizes the Q-learning function
        env: environment that the rl agent interacts with
        leraning_rate_q (float): 
        leraning_rate_policy (float): 
        weight_decay (float): regularization 
        target_learning_rate (float): 
        batch_size (int): 
        exploration_rate (float): 
        exploration_decay (float): 
        exploration_min (float): 
        num_episodes (int): maximum number of episodes
    Returns: 
        scores (numpy.ndarray): the rewards of each episode
        actions (numpy.ndarray): the actions chosen by the agent
    """
    actor_target_net = deepcopy(actor_net)
    critic_target_net = deepcopy(critic_net)
    optimizer_actor = optim.Adam(actor_net.parameters(), lr=alpha_actor, weight_decay=weight_decay)
    optimizer_critic = optim.Adam(critic_net.parameters(), lr=alpha_critic, weight_decay=weight_decay)
    replay_buffer = ReplayMemory(1000)
    reward_history = []
    action_history = []

    if not train:
        exploration_rate = exploration_min
    
    for n in range(num_episodes):
        rewards = []
        actions = []
        state = env.reset() 

        for i in range(max_episode_length):
            action = act(actor_net, state, exploration_rate) 
            next_state, reward, done, _ = env.step(action) 

            if done:
                if train:
                    update(replay_buffer, max(2, (i%batch_size)), critic_net, critic_target_net, actor_net, actor_target_net, optimizer_critic, optimizer_actor)
                break

            actions.append(action)
            rewards.append(reward)
            replay_buffer.push(torch.from_numpy(state).float().unsqueeze(0).to(device), 
                            torch.FloatTensor([action]), 
                            torch.FloatTensor([reward]), 
                            torch.from_numpy(next_state).float().unsqueeze(0).to(device))

            if train and len(replay_buffer) >= batch_size and i % batch_size == 0: 
                update(replay_buffer, batch_size, critic_net, critic_target_net, actor_net, actor_target_net, optimizer_critic, optimizer_actor)
                soft_updates(critic_net, critic_target_net, target_learning_rate)
                soft_updates(actor_net, actor_target_net, target_learning_rate)

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
