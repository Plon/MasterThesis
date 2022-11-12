import numpy as np
import torch
torch.manual_seed(0)
import torch.optim as optim


def optimize(optimizer, loss) -> None: 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def reinforce(policy_network, env, alpha=1e-3, weight_decay=1e-5, num_episodes=np.iinfo(np.int32).max, train=True) -> tuple[np.ndarray, np.ndarray]: 
    """
    Online Monte Carlo policy gradient 
    Every trajectory consists of one step, and then the loss is computed

    Args: 
        policy_network (nn.Model): network that parameterizes the policy
        env: environment that the rl agent interacts with
        alpha (float): learning rate
        weight_decay (float): regularization 
        num_episodes (int): maximum number of episodes
    Returns: 
        scores (numpy.ndarray): the rewards of each episode
        actions (numpy.ndarray): the actions chosen by the agent
    """
    optimizer = optim.Adam(policy_network.parameters(), lr=alpha, weight_decay=weight_decay)
    rewards = [] 
    actions = [] 
    state = env.state() #S_0

    for _ in range(num_episodes):
        action, log_prob = policy_network.act(state) #A_{t-1}
        state, reward, done, _ = env.step(action) # S_t, R_t 

        if done:
            break

        actions.append(action)
        rewards.append(reward) 

        if train:
            loss = - log_prob * reward
            optimize(optimizer, loss)

    return np.array(rewards), np.array(actions)
