import numpy as np
import torch
torch.manual_seed(0)
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from reinforce import optimize
criterion = torch.nn.MSELoss()


def recurrent_reinforce_baseline(policy_network, value_function, env, alpha_policy=1e-3, alpha_vf=1e-5,  weight_decay=1e-5, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, train=True, print_res=True, print_freq=100) -> tuple[np.ndarray, np.ndarray]: 
    optimizer_policy = optim.Adam(policy_network.parameters(), lr=alpha_policy, weight_decay=weight_decay)
    optimizer_vf = optim.Adam(value_function.parameters(), lr=alpha_vf, weight_decay=weight_decay)
    reward_history = []
    action_history = []

    if not train:
        policy_network.eval()
        value_function.eval()
    else:
        policy_network.train()
        value_function.train()

    for n in range(num_episodes):
        state = env.reset() #S_0
        hx = None #h_0
        rewards = [] 
        actions = [] 
        log_probs = []  
        states = []

        for _ in range(max_episode_length):
            action, log_prob, hx = policy_network.act(state, hx) #A_{t-1}
            state, reward, done, _ = env.step(action) # S_t, R_t 

            if done:
                break

            actions.append(action)
            rewards.append(reward) 
            log_probs.append(log_prob)
            states.append(torch.from_numpy(state).float().unsqueeze(0).to(device))

        if train:
            r = torch.FloatTensor(rewards)
            #r = (r - r.mean()) / (r.std() + float(np.finfo(np.float32).eps))
            state_value = value_function(torch.cat(states)).squeeze()
            adv = r - state_value.detach()            
            adv = (adv - adv.mean()) / (adv.std() + float(np.finfo(np.float32).eps))

            log_probs = torch.stack(log_probs).squeeze()
            policy_loss = torch.mul(log_probs, adv).mul(-1).sum()
            optimize(optimizer_policy, policy_loss)

            vf_loss = criterion(state_value, r)
            optimize(optimizer_vf, vf_loss)
        
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