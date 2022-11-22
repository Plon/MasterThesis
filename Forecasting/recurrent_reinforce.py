import numpy as np
import torch
torch.manual_seed(0)
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from reinforce import optimize, get_policy_loss


def recurrent_reinforce(policy_network, env, alpha=1e-3, weight_decay=1e-5, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, train=True, print_res=True, print_freq=100) -> tuple[np.ndarray, np.ndarray]: 
    optimizer = optim.Adam(policy_network.parameters(), lr=alpha, weight_decay=weight_decay)
    reward_history = []
    action_history = []

    if not train:
        policy_network.eval()
    else:
        policy_network.train()

    for n in range(num_episodes):
        state = env.reset() #S_0
        hx = None #h_0
        rewards = [] 
        actions = [] 
        log_probs = []  

        for _ in range(max_episode_length):
            action, log_prob, hx = policy_network.act(state, hx) #A_{t-1}
            state, reward, done, _ = env.step(action) # S_t, R_t 

            if done:
                break

            actions.append(action)
            rewards.append(reward) 
            log_probs.append(log_prob)

        if train:
            policy_loss = get_policy_loss(rewards, log_probs)
            optimize(optimizer, policy_loss)
        
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
