import numpy as np
import torch
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#TODO very sensitive to hyperparameters - find better hyperparameters
# Actor-Critic with eligibility traces, continuing (undiscounted) setting
def actor_critic(actor_net, critic_net, env, lambda_actor=1e-1, lambda_critic=1e-1, alpha_actor=1e-1, alpha_critic=1e-3, alpha_R_hat=1e-1,  num_episodes=np.iinfo(np.int32).max, train=True) -> tuple[np.ndarray, np.ndarray]: 
    rewards = []
    actions = []
    R_hat = 0
    state = env.state()
    actor_trace = []
    critic_trace = []

    for _ in range(num_episodes):
        action, log_prob = actor_net.act(state)
        new_state, reward, done, _ = env.step(action)

        if done:
            break

        actions.append(action)
        rewards.append(reward)

        if train: 
            #Calculate eligibility trace of actor net
            actor_net.zero_grad()
            log_prob.backward(retain_graph=True)
            if not actor_trace:
                with torch.no_grad():
                    for p in actor_net.parameters():
                        actor_trace.append(p.grad)
            else: 
                with torch.no_grad():
                    for i, p in enumerate(actor_net.parameters()):
                        actor_trace[i] = actor_trace[i] * lambda_actor + p.grad

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)        
            state_tensor.requires_grad = True
            state_val = critic_net(state_tensor)

            #Calculate eligibility trace of critic net
            critic_net.zero_grad()
            state_val.backward()
            if not critic_trace:
                with torch.no_grad():
                    for p in critic_net.parameters():
                        critic_trace.append(p.grad)
            else: 
                with torch.no_grad():
                    for i, p in enumerate(critic_net.parameters()):
                        critic_trace[i] = critic_trace[i] * lambda_critic + p.grad

            new_state_tensor = torch.from_numpy(new_state).float().unsqueeze(0).to(device) 
            new_state_val = critic_net(new_state_tensor)

            advantage = reward - R_hat + new_state_val.item() - state_val.item()
            R_hat += alpha_R_hat * advantage

            #perform step in actor net using eligibility trace
            actor_net.zero_grad()
            with torch.no_grad():
                for i, p in enumerate(actor_net.parameters()):
                    new_val = p + alpha_actor * advantage * actor_trace[i]
                    p.copy_(new_val)

            #perform step in critic net using eligibility trace
            critic_net.zero_grad()
            with torch.no_grad():
                for i, p in enumerate(critic_net.parameters()):
                    new_val = p + alpha_critic * advantage * critic_trace[i]
                    p.copy_(new_val)

        state = new_state

    return np.array(rewards), np.array(actions)
