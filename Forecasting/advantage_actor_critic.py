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
criterion = torch.nn.MSELoss()


def update(replay_buffer: ReplayMemory, batch_size: int, actor: torch.nn.Module, critic: torch.nn.Module, critic_target: torch.nn.Module, optimizer_actor: torch.optim, optimizer_critic: torch.optim, discrete=True) -> None: 
    batch = get_batch(replay_buffer, batch_size)
    if batch is None:
        return
    state_batch, action_batch, reward_batch, next_state_batch = batch
    action_batch = action_batch + 1 #need to do because i -1 actions
    reward_batch = (reward_batch - reward_batch.mean()) / (reward_batch.std() + float(np.finfo(np.float32).eps))

    log_probs = get_log_probs(actor, state_batch, action_batch, discrete)
    actor_loss, critic_loss = get_actor_and_critic_loss(critic, critic_target, state_batch, reward_batch, next_state_batch, log_probs)

    #advantage = get_advantage(critic, critic_target, state_batch, reward_batch, next_state_batch)
    #advantage = (advantage - advantage.mean()) / (advantage.std() + float(np.finfo(np.float32).eps))

    #critic_loss = advantage.pow(2).mean()
    optimize(optimizer_critic, critic_loss)

    #log_probs = get_log_probs(actor, state_batch, action_batch, discrete)
    #actor_loss = (-log_probs * advantage.detach()).mean()
    optimize(optimizer_actor, actor_loss)


def get_actor_and_critic_loss(critic, critic_target, state_batch, reward_batch, next_state_batch, log_probs) -> tuple[torch.Tensor, torch.Tensor]:
    state_val = critic(state_batch).squeeze()
    with torch.no_grad():
        #next_state_val = critic(next_state_batch).squeeze()
        next_state_val = critic_target(next_state_batch).squeeze()
    advantage = reward_batch + next_state_val - state_val

    actor_loss = (-log_probs * advantage.detach()).mean() 

    #print(reward_batch, next_state_val, state_val)
    critic_loss = criterion((reward_batch+next_state_val), state_val)
    #print(actor_loss, critic_loss)
    return actor_loss, critic_loss


def get_advantage(critic_net, critic_target, state, reward, next_state) -> torch.Tensor:
    state_val = critic_net(state).squeeze()
    with torch.no_grad():
        #next_state_val = critic_net(next_state).squeeze()
        next_state_val = critic_target(next_state).squeeze()
    advantage = reward + next_state_val - state_val
    return advantage 


def get_log_probs(actor, state, action, discrete=True) -> torch.Tensor:
    if discrete:
        dist, _ = actor(state)
        dist = F.softmax(dist, dim=1)
        dist = Categorical(dist) 
    else:
        mean, std, _ = actor(state)
        mean = torch.clamp(mean, -1, 1) + 1. ##need to do this since the discrete outputs {0, 1, 2}
        std += 1e-8
        dist = Normal(mean, std)
        #print("Action:", action)
        #print("mean:", mean)
        #print("std:", std)
        #print("-"*20, "\n")
    log_probs = dist.log_prob(action)
    return log_probs


def advantage_actor_critic(actor_net, critic_net, env, alpha_actor=1e-1, alpha_critic=1e-3, weight_decay=1e-6, target_critic_net_learning_rate=0.5, batch_size=10, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, train=True, print_res=True, print_freq=100, discrete=True) -> tuple[np.ndarray, np.ndarray]: 
    """ Trains the actor-critic with eligibility traces in the continuing undiscounted setting """
    critic_target_net = deepcopy(critic_net) #not used, works better without
    critic_target_net.eval()
    optimizer_actor = optim.Adam(actor_net.parameters(), lr=alpha_actor, weight_decay=weight_decay)
    optimizer_critic = optim.Adam(critic_net.parameters(), lr=alpha_critic, weight_decay=weight_decay)
    reward_history = []
    action_history = []
    batch_size = max(2, batch_size) ###
    replay_buffer = ReplayMemory(1000) # should this be initialized before every episode?

    if not train:
        actor_net.eval()
        critic_net.eval()
    else:
        actor_net.train()
        critic_net.train()

    for n in range(num_episodes):
        rewards = []
        actions = []
        hx = None
        state = env.reset()

        for i in range(max_episode_length): 
            action, _, hx = actor_net.act(state, hx)
            next_state, reward, done, _ = env.step(action)

            if done:
                #if train:
                #    update(replay_buffer, min(batch_size, i), actor_net, critic_net, critic_target_net, optimizer_actor, optimizer_critic, discrete)
                break

            actions.append(action)
            rewards.append(reward)
            replay_buffer.push(torch.from_numpy(state).float().unsqueeze(0).to(device), 
                            torch.FloatTensor([action]), 
                            torch.FloatTensor([reward]), 
                            torch.from_numpy(next_state).float().unsqueeze(0).to(device))

            if train and len(replay_buffer) >= batch_size and (i+1) % min(max(int(batch_size/10), 2), max_episode_length) == 0:
                update(replay_buffer, batch_size, actor_net, critic_net, critic_target_net, optimizer_actor, optimizer_critic, discrete)

            soft_updates(critic_net, critic_target_net, target_critic_net_learning_rate) ### not used
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
