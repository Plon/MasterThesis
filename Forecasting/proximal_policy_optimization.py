import numpy as np
import torch
torch.manual_seed(0)
import torch.optim as optim
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from batch_learning import ReplayMemory, Transition, get_batch
from reinforce import optimize
from deep_q_network import compute_loss_dqn
from deep_deterministic_policy_gradient import compute_critic_loss
epsilon = 5e-1 # add as hyperparameter


def update(replay_buffer: ReplayMemory, batch_size: int, critic: torch.nn.Module, actor: torch.nn.Module, optimizer_critic: torch.optim, optimizer_actor: torch.optim, act, recurrent=False) -> None: 
    """ The idea is to calculate policy loss and critic loss and optimize. 
        However, the algorithm doesn't fit the problem at all. 
        It uses advantage function for state-value function, but talking about state values 
        in this context makes little sense, and therefore the advantage gives no information. 
        I have therefore chosen to use a state-action function, but that cannot be used to get
        advantage estimate. I just use the state-action value as the advantage, but that is 
        not optimal. Also to get this to work well with other problems there would need to be
        some additional argument that would return logprobs. This solution is hard-coded to (only) 
        work for single instrument trading with discrete action space. """
    batch = get_batch(replay_buffer, batch_size)
    if batch is None:
        return
    state, action, reward, selected_prob = batch

    if recurrent:
        state_action_val = critic(state, action.view(action.shape[0], -1))[0].squeeze()
    else:
        state_action_val = critic(state, action.view(action.shape[0], -1)).squeeze()
    delta = state_action_val.detach()
    delta = (delta - delta.mean()) / (delta.std() + float(np.finfo(np.float32).eps))

    ######################
    # This part only works for single instrument discrete trading atm
    # If it is to work with other variants make functions get_logprobs and pass them as args to the ppo algo
    if recurrent: 
        x, _ = actor(state)
    else:
        x = actor(state)
    probs = nn.functional.softmax(x, dim=1)
    m = torch.distributions.Categorical(probs) 
    a = m.sample() 
    log_probs = m.log_prob(a)
    ######################

    ppo_loss = compute_ppo_loss(log_probs, selected_prob, delta)
    optimize(optimizer_actor, ppo_loss)    

    c_loss = compute_critic_loss(critic, batch)
    optimize(optimizer_critic, c_loss)


def compute_ppo_loss(log_probs, selected_prob, delta) -> torch.Tensor: 
    prob_ratio = torch.exp(log_probs - selected_prob) #e^(log x - log y) = x/y
    a = prob_ratio * delta
    b = torch.clamp(prob_ratio, 1 - epsilon, 1 + epsilon) * delta
    ppo_loss = (-torch.min(a, b)).mean()
    return ppo_loss


def proximal_policy_optimization(actor_net, critic_net, env, act, alpha_actor=1e-3, alpha_critic=1e-3, weight_decay=1e-4, batch_size=30, update_freq=1, num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, train=True, print_res=True, print_freq=100, recurrent=False) -> tuple[np.ndarray, np.ndarray]: 
    optimizer_actor = optim.Adam(actor_net.parameters(), lr=alpha_actor, weight_decay=weight_decay)
    optimizer_critic = optim.Adam(critic_net.parameters(), lr=alpha_critic, weight_decay=weight_decay)
    replay_buffer = ReplayMemory(batch_size)
    reward_history = []
    action_history = []

    if not train:
        actor_net.eval()
        critic_net.eval()
    else:
        actor_net.train()
        critic_net.train()
        
    for n in range(num_episodes):
        rewards = []
        actions = []
        state = env.reset() 
        hx = None

        for i in range(max_episode_length):
            action, log_prob, hx = act(actor_net, state, hx, recurrent) 
            next_state, reward, done, _ = env.step(action) 

            if done:
                break

            actions.append(action)
            rewards.append(reward)
            replay_buffer.push(torch.from_numpy(state).float().unsqueeze(0).to(device), 
                            torch.FloatTensor([action]), 
                            torch.FloatTensor([reward]), 
                            log_prob.detach())

            if train and len(replay_buffer) >= batch_size and (i+1) % update_freq == 0:
                update(replay_buffer, batch_size, critic_net, actor_net, optimizer_critic, optimizer_actor, act, recurrent)    
            
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
