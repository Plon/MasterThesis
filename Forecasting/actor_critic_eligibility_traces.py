import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActorNetworkContinuous(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=1, action_bounds=1) -> None:
        super(ActorNetworkContinuous, self).__init__()
        self.action_bounds = action_bounds
        self.input_layer = nn.Linear(observation_space, hidden_size)
        self.output_layer = nn.Linear(hidden_size, action_space)

        logstds_param = nn.Parameter(torch.full((action_space,), 0.1))
        self.register_parameter("logstds", logstds_param)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.input_layer(x))
        return self.output_layer(x)
    
    def act(self, state) -> tuple[float, torch.Tensor]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        mean = self.forward(state).cpu()
        std = torch.clamp(self.logstds.exp(), 1e-3, 50)
        dist = Normal(mean, std) 
        action = dist.sample() 
        return torch.clamp(action, -self.action_bounds, self.action_bounds).item(), dist.log_prob(action)


class ActorNetworkDiscrete(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128, action_space=3):
        super(ActorNetworkDiscrete, self).__init__()
        self.input_layer = nn.Linear(observation_space, hidden_size)
        self.output_layer = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.output_layer(x)
        return F.softmax(x, dim=1)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()        
        m = Categorical(probs) 
        action = m.sample() 
        return (action.item() - 1), m.log_prob(action)


class CriticNetwork(nn.Module):
    def __init__(self, observation_space=8, hidden_size=128) -> None:
        super(CriticNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        return self.output_layer(x)


#TODO find better hyperparameters
# Actor-Critic with eligibility traces, continuing (undiscounted) setting
def actor_critic(actor_net, critic_net, env, lambda_actor=0.1, lambda_critic=0.1, alpha_actor=1e-2, alpha_critic=1e-2, alpha_R_hat=1e-2,  num_episodes=np.iinfo(np.int32).max): 
    scores = []
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
        scores.append(reward)

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

    return np.array(scores), np.array(actions)
