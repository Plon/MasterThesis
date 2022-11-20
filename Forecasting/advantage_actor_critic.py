from copy import deepcopy
import numpy as np
import torch
torch.manual_seed(0)
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from batch_learning import ReplayMemory, Transition, get_batch
from deep_deterministic_policy_gradient import soft_updates
from reinforce import optimize
nn_activation_function = torch.nn.LeakyReLU()


class A2CLSTMActorContinuous(torch.nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, num_lstm_layers=1) -> None:
        super(A2CLSTMActorContinuous, self).__init__()
        self.input_layer = torch.nn.Linear(observation_space, hidden_size)
        self.lstm_layer = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=True)
        self.mean_layer = torch.nn.Linear(hidden_size, 1)        
        self.std_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, hx=None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_layer(x)
        x = nn_activation_function(x)        
        if hx is not None:
            x, hx = self.lstm_layer(x, hx)
        else: 
            x, hx = self.lstm_layer(x)
        
        mean = torch.tanh(self.mean_layer(x))
        std = F.softplus(self.std_layer(x))

        mean = torch.clamp(mean, -1, 1) + 1. ##need to do this since the discrete outputs {0, 1, 2}
        std += 1e-5
        dist = Normal(mean, std)

        return dist, hx

    def act(self, state, hx=None) -> tuple[int, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        m, hx = self.forward(state, hx)     
        action = m.sample() 
        return (torch.clamp(action, 0, 2).item() - 1.), hx


class A2CLSTMActorDiscrete(torch.nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3, num_lstm_layers=1) -> None:
        super(A2CLSTMActorDiscrete, self).__init__()
        self.input_layer = torch.nn.Linear(observation_space, hidden_size)
        self.lstm_layer = torch.nn.LSTM(input_size=hidden_size, hidden_size=action_space, num_layers=num_lstm_layers, batch_first=True)

    def forward(self, x, hx=None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_layer(x)
        x = nn_activation_function(x)        
        if hx is not None:
            x, hx = self.lstm_layer(x, hx)
        else: 
            x, hx = self.lstm_layer(x)
        probs = F.softmax(x, dim=1)
        dist = Categorical(probs)
        return dist, hx

    def act(self, state, hx=None) -> tuple[int, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        m, hx = self.forward(state, hx)      
        action = m.sample() 
        return (action.item() - 1), hx


def update(replay_buffer: ReplayMemory, batch_size: int, actor: torch.nn.Module, critic: torch.nn.Module, critic_target: torch.nn.Module, optimizer_actor: torch.optim, optimizer_critic: torch.optim) -> None: 
    batch = get_batch(replay_buffer, batch_size)
    if batch is None:
        return 0
    state_batch, action_batch, reward_batch, next_state_batch = batch
    action_batch = action_batch + 1 #need to do because i -1 actions

    advantage = get_advantage(critic, critic_target, state_batch, reward_batch, next_state_batch)
    #advantage = (advantage - advantage.mean()) / (advantage.std() + float(np.finfo(np.float32).eps))

    critic_loss = advantage.pow(2).mean()
    optimize(optimizer_critic, critic_loss)

    log_probs = get_log_probs(actor, state_batch, action_batch)
    actor_loss = (-log_probs * advantage.detach()).mean()
    optimize(optimizer_actor, actor_loss)


def get_advantage(critic_net, critic_target, state, reward, next_state) -> torch.Tensor:
    state_val = critic_net(state).squeeze()
    with torch.no_grad():
        next_state_val = critic_net(next_state).squeeze()
        #next_state_val = critic_target(next_state).squeeze()
    advantage = reward + next_state_val - state_val
    return advantage 


def get_log_probs(actor, state, action) -> torch.Tensor:
    dist, _ = actor(state)
    log_probs = dist.log_prob(action)
    return log_probs


def advantage_actor_critic(actor_net, critic_net, env, alpha_actor=1e-1, alpha_critic=1e-3, weight_decay=1e-6, target_critic_net_learning_rate=0.5, batch_size=10, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, train=True, print_res=True, print_freq=100) -> tuple[np.ndarray, np.ndarray]: 
    """ Trains the actor-critic with eligibility traces in the continuing undiscounted setting """
    critic_target_net = deepcopy(critic_net) #not used, works better without
    optimizer_actor = optim.Adam(actor_net.parameters(), lr=alpha_actor, weight_decay=weight_decay)
    optimizer_critic = optim.Adam(critic_net.parameters(), lr=alpha_critic, weight_decay=weight_decay)
    reward_history = []
    action_history = []

    batch_size = max(2, batch_size) ###
    replay_buffer = ReplayMemory(1000) #TODO should this be initialized before every episode?

    for n in range(num_episodes):
        rewards = []
        actions = []
        hx = None
        state = env.reset()

        for i in range(max_episode_length): 
            action, hx = actor_net.act(state, hx)
            next_state, reward, done, _ = env.step(action)

            if done:
                if train:
                    batch_size = max(2, (i % batch_size))
                    update(replay_buffer, batch_size, actor_net, critic_net, critic_target_net, optimizer_actor, optimizer_critic)
                break

            actions.append(action)
            rewards.append(reward)
            replay_buffer.push(torch.from_numpy(state).float().unsqueeze(0).to(device), 
                            torch.FloatTensor([action]), 
                            torch.FloatTensor([reward]), 
                            torch.from_numpy(next_state).float().unsqueeze(0).to(device))

            if train:
                if (i+1) % batch_size == 0:
                    update(replay_buffer, batch_size, actor_net, critic_net, critic_target_net, optimizer_actor, optimizer_critic)
                    soft_updates(critic_net, critic_target_net, target_critic_net_learning_rate)
    
            state = next_state
        
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
