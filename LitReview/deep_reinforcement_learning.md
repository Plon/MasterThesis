# Deep Reinforcement Learning. 
Deep reinforcement learning is a research field that combines Deep Learning (DL) and reinforcement learning. 
In deep RL, one or more of the value function, action-value function, or policy is approximated as an artificial neural network. 
Recent advances in deep learning have made it possible to extract high-level features from unprocessed data \cite{goodfellow2016deep}. 
Deep RL aims to be able to control agents directly from high-dimensional sensory input like vision or sound, without the need for manual feature engineering \cite{li2017deep}.
Combining deep learning and reinforcement learning can be challenging. 
DL generally requires labeled training data, while RL learns using delayed reward signals. 
Deep learning typically assumes that data is independent and identically distributed (IID), while RL assumes that states are highly correlated. 
Deep learning also assumes a fixed underlying data distribution, while RL can change as the policy evolves. 
The combination of nonlinear function approximation and bootstrapping may lead to unstable learning and divergence \cite{sutton2018reinforcement}. 



## Deep Q Network. 
Deep Q Network (DQN) is a model-free value-based RL algorithm that extends the tabular Q-learning algorithm to the continuous state space. 
It was introduced by researchers from DeepMind in 2013 \cite{mnih2013playing}. 
DQN approximates the Q-values using a convolutional neural network instead of a lookup table like Q-learning. 
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






## Deep Recurrent Q-Network. 
DQN uses the last four frames as input to its convolutional neural network. 
The problem becomes non-Markovian if the future states depend on more than the DQN's current input. 
Thus, it becomes a Partially-Observable Markov Decision Process (POMDP). 
Deep Recurrent Q-Network (DQRN) adds recurrency to deep Q-networks by replacing the first post-convolutional fully-connected layer with a recurrent LSTM. 
It was introduced in 2015 \cite{hausknecht2015deep}. 
Instead of stacking a history of frames as DQN does, the DQRN uses reccurency to be able to match DQN's performance while only seeing one frame at a time. 
DRQN handles partial observability. It can generalize its policies to complete observations when trained with partial observations, and vice-versa. 
Although recurrency is a viable alternative to stacking observations, it is not assumed to improve performance. 







## Deep Deterministic Policy Gradients. 
Deep Deterministic Policy Gradients (DDPG) is an extension of the DQN algorithm that combines Q-learning with actor-critic to create a model-free algorithm with continuous action space where the critic learns a Q-function and the actor learns a deterministic parameterized policy based on the Q-values from the critic. 
It was introduced by researchers from DeepMind in 2015 \cite{lillicrap2015continuous}. 


In DDPG, the critic is represented by the Q-network and target Q-network, just as in DQN. The Q-network is updated in the same way as for DQN. 
In addition, two actor neural nets are employed in DDPG. 
The actor is represented by a deterministic policy function and a target policy network. 
In the same way as the target Q-network, the target policy is a time-delayed copy of the original network. 
DDPG's target networks are time-delayed copies of the original, like DQN. However, weights $\theta'$ slowly track the learned weights $\theta$: $\theta' \leftarrow \tau \theta + (1-\tau) \theta'$, using a positive learning rate $\tau \ll 1$.
The policy function $\mu_{\theta^\mu}(s)$ deterministically maps states to actions and not a probability distribution across an action space as actor-critic methods like A2C. 
For the policy network, the objective is to maximize expected return, represented by the objective function $J(\theta)=\mathbb{E}[Q_{\theta^Q}(s,a) | s=s_t,a=\mu_{\theta^\mu}(s_t)]$. 
The policy network is updated using stochastic gradient descent on the stochastic estimate
\begin{equation}
\nabla_{\theta^\mu} J(\theta) \approx \nabla_a Q_{\theta^Q}(s,a) \nabla_{\theta^\mu} \mu_{\theta^\mu}(s)
\end{equation}


DDPG uses the experience replay technique to minimize correlations between samples. 
It is common to add noise to actions to explore a continuous action space. 
For DDPG this is done using the Ornstein-Uhlenbeck Process \cite{uhlenbeck1930theory} that adds noise that is temporally correlated. 


