# Deep Reinforcement Learning. 
Deep reinforcement learning combines Deep Learning (DL) and reinforcement learning. 
It aims to be able to control agents directly from high-dimensional sensory input like vision or sound, without the need for manual feature engineering. 
Recent advances in deep learning have made it possible to extract high-level features from unprocessed data. 
Combining deep learning and reinforcement learning can be challenging as DL generally requires labeled training data, while RL learns using reward signals that can be noisy and delayed. 
Deep learning typically assumes that data is independent and identically distributed, while RL assumes states are highly correlated. 
Deep learning also assumes a fixed underlying data distribution, while RL can change as the policy evolves. 




## Deep Q Network. 
Deep Q Network (DQN) is a model-free value-based RL algorithm that extends the tabular Q-learning algorithm to the continuous state space. 
It was introduced by researchers from DeepMind in 2013 \cite{mnih2013playing}. 
DQN approximates the Q-values using artificial neural networks instead of a lookup table like Q-learning. 
The network takes the state as input and outputs the Q-values of all actions from that state. 
For this reason, DQN only works for low-dimensional, discrete action spaces. 
As for Q-learning, the Q-values are initialized to random estimates and then improved as the environment is explored. The algorithm is off-policy as it learns the greedy policy while following a non-greedy policy, e.g., $\epsilon$-greedy. 
The Q-network is a neural network, a nonlinear function approximation, with weights $\theta$ that is trained by minimizing the mean squared error of the objective function $L(\theta)$ 
\begin{equation}
L(\theta) = \mathbb{E} [(Q'(S,A) - Q_theta(S,A))^2]
\end{equation}
where $Q'(S_t,A_t) = R_{t+1} + \gamma \argmax_{A'} Q_\theta (S_{t+1}, A')$. 
The weights $\theta$ are updated using stochastic gradient ascent. 



The DQN is able to learn action-value functions using large, nonlinear function approximation techniques in a stable and robust manner for two reasons. 
Firstly, it trains the network off-policy with samples from a replay buffer to minimize correlations between samples. 
Firstly, it trains the network off-policy with samples from an experience replay buffer. 
Experience replay stores experiences $e_t = (s_t, a_t, r_t, s_{t+1})$ into a replay memory that the agent draws from at random when updating Q-values. 
This technique minimizes correlations between samples. 
Secondly, DQN solves the problem of “chasing tails”. 
The Q-network is used to generate both the predicted value and target value, and the weights are updated after each iteration.
Changing the weights, however, will also change the target values, so they are essentially moving targets. 
DQN solves this by maintaining a target Q-network, identical to the Q-network, in parallel to ensure consistency during TD backups. 
The target Q-network copies the learned weights of the Q-network after some predefined number of iterations. 





## Deep Deterministic Policy Gradients. 
Deep Deterministic Policy Gradients (DDPG) is an extension of the DQN algorithm that combines Q-learning with actor-critic to create a model-free algorithm with continuous action space where the critic learns a Q-function and the actor learns a deterministic parameterized policy based on the Q-values from the critic. 
It was introduced by researchers from DeepMind in 2016 \cite{lillicrap2015continuous}. 


In DDPG, the critic is represented by the Q-network and target Q-network, just as in DQN. The Q-network is updated in the same way as for DQN. 
In addition, two actor neural nets are employed in DDPG. 
The actor is represented by a deterministic policy function and a target policy network. 
In the same way as the target Q-network, the target policy is a time-delayed copy of the original network. 
The policy function $\mu_{\theta^\mu}(s)$ deterministically maps states to actions and not a probability distribution across an action space as actor-critic methods like A2C. 
For the policy network, the objective is to maximize expected return, represented by the objective function $J(\theta)=\mathbb{E}[Q_{\theta^Q}(s,a) | s=s_t,a=\mu_{\theta^\mu}(s_t)]$. 
The policy network is updated using stochastic gradient descent on the stochastic estimate
\begin{equation}
\nabla_{\theta^\mu} J(\theta) \approx \nabla_a Q_{\theta^Q}(s,a) \nabla_{\theta^\mu} \mu_{\theta^\mu}(s)
\end{equation}


DDPG uses the experience replay technique to minimize correlations between samples. 
It is common to add noise to actions to explore a continuous action space. 
For DDPG this is done using the Ornstein-Uhlenbeck Process \cite{uhlenbeck1930theory} that adds noise that is temporally correlated. 


