import numpy as np
import torch
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def update_eligibiltiy_trace(eligibility_traces: list, net: torch.nn.Module, lambd: float, loss: torch.Tensor, retain_graph=False) -> None:
    """ Compute eligibility traces for neural net """
    net.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if not eligibility_traces:
        with torch.no_grad():
            for p in net.parameters():
                eligibility_traces.append(p.grad)
    else: 
        with torch.no_grad():
            for i, p in enumerate(net.parameters()):
                eligibility_traces[i] = eligibility_traces[i] * lambd + p.grad


def optimize(eligibility_traces: list, net: torch.nn.Module, advantage: float, alpha: float) -> None:
    """ Optimize parameters of the neural net using eligibility traces """
    net.zero_grad()
    with torch.no_grad():
        for i, p in enumerate(net.parameters()):
            p.add_(alpha * advantage * eligibility_traces[i])


def get_state_vals(critic_net: torch.nn.Module, state: np.ndarray, new_state: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """ Returns state val of current and next state """
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)        
    state_tensor.requires_grad = True
    state_val = critic_net(state_tensor)

    new_state_tensor = torch.from_numpy(new_state).float().unsqueeze(0).to(device) 
    new_state_val = critic_net(new_state_tensor)

    return state_val, new_state_val


def update(actor_net: torch.nn.Module, critic_net: torch.nn.Module, actor_trace: list, critic_trace: list, log_prob: torch.Tensor, reward: torch.Tensor, lambda_actor: float, lambda_critic: float, alpha_actor: float, alpha_critic: float, R_hat: float, state: np.ndarray, new_state: np.ndarray) -> torch.Tensor:
    """ Computes advantage function, updates eligibility traces, optimizes neural nets """
    state_val, new_state_val = get_state_vals(critic_net, state, new_state)
    advantage = reward - R_hat + new_state_val.item() - state_val.item()

    update_eligibiltiy_trace(actor_trace, actor_net, lambda_actor, log_prob, True)
    update_eligibiltiy_trace(critic_trace, critic_net, lambda_critic, state_val)
    
    optimize(actor_trace, actor_net, advantage, alpha_actor)
    optimize(critic_trace, critic_net, advantage, alpha_critic)
    
    return advantage


def actor_critic(actor_net, critic_net, env, lambda_actor=1e-1, lambda_critic=1e-1, alpha_actor=1e-1, alpha_critic=1e-3, alpha_R_hat=1e-1,  num_episodes=1000, max_episode_length=np.iinfo(np.int32).max, train=True, print_res=True, print_freq=100) -> tuple[np.ndarray, np.ndarray]: 
    """ Trains the actor-critic with eligibility traces in the continuing undiscounted setting """
    reward_history = []
    action_history = []
    
    for n in range(num_episodes):
        rewards = []
        actions = []
        R_hat = 0
        state = env.reset()
        actor_trace = []
        critic_trace = []

        for _ in range(max_episode_length):
            action, log_prob = actor_net.act(state)
            next_state, reward, done, _ = env.step(action)

            if done:
                break

            actions.append(action)
            rewards.append(reward)

            if train:
                advantage = update(actor_net, critic_net, actor_trace, critic_trace, log_prob, reward, lambda_actor, lambda_critic, alpha_actor, alpha_critic, R_hat, state, next_state)
                R_hat += alpha_R_hat * advantage

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
