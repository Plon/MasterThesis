import numpy as np
import torch
torch.manual_seed(0)
import torch.optim as optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.MSELoss()


def optimize(optimizer, loss) -> None: 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def reinforce_baseline(policy_network, value_function, env, alpha_policy=1e-3, alpha_vf=1e-5, weight_decay=1e-5, num_episodes=np.iinfo(np.int32).max, train=True) -> tuple[np.ndarray, np.ndarray]: 
    optimizer_policy = optim.Adam(policy_network.parameters(), lr=alpha_policy, weight_decay=weight_decay)
    optimizer_vf = optim.Adam(value_function.parameters(), lr=alpha_vf, weight_decay=weight_decay)
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
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            state_value = value_function(state_tensor).squeeze()
            advantage = reward - state_value.detach()
            policy_loss = - log_prob * advantage
            optimize(optimizer_policy, policy_loss)

            reward_tensor = torch.FloatTensor([reward]).squeeze()
            vf_loss = criterion(state_value, reward_tensor)
            optimize(optimizer_vf, vf_loss)

    return np.array(rewards), np.array(actions)
