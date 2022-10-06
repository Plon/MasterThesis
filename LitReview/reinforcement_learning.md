# Machine Learning 
Machine Learning (ML) is one of the most prominent research areas in information technology (IT), and a central part of modern life. 
It is the field of study that gives computers the ability to learn without being explicitly programmed \cite{marsland2011machine}. 
The field grew out of Artificial Intelligence (AI) in the late 1970s.
ML has significant real-world applications like computer vision, speech recognition, and robot control.
It is also helping accelerate discoveries in empirical sciences like psychology, biology, and economics \cite{mitchell2006discipline}. 
The goal of machine learning is to build computer systems that learn from experience and automatically improve. 


In essence, machine learning attempts to extract generalizable predictive patterns from data. 
ML is closely related to statistics in the way it attempts to transform raw data into knowledge. 
However, while statistical models are designed for inference, ML models are designed for prediction. 
It is at its core an optimization problem where performance improves through leveraging data. 
Generalizability is a crucial aspect of ML. Models should be capable of transferring the learned patterns to new previously unseen samples without sacrificing accuracy \cite{marsland2011machine}. 
It is therefore important that the distribution of the data used to train the model is representative of the entire population. 


In situations where the application is too complex for humans to manually design the algorithm, or the software has to be customized to the operational environment after deployment, machine learning is an ideal solution \cite{mitchell2006discipline}. 
There are three primary ML paradigms; Supervised Learning, Unsupervised Learning, and Reinforcement Learning. 






# Reinforcement Learning
Reinforcement Learning (RL) is a Machine Learning approach where an agent observes an environment, and takes actions based on the state of the environment. 
Actions influence immediate rewards and the subsequent state of the environment and thus future rewards. 
The agent refines a policy to maximize the total rewards it receives. 
This agent-environment interaction is often modeled as a Markov Decision Process (MDP). 


RL has two distinguishing features from other ML approaches: trial-and-error search and delayed reward \cite{sutton2018reinforcement}. 
Delayed reward refers to the previously mentioned fact that the agent's actions determine the next state and thus future rewards. Therefore, an agent might perform an action that will be valuable at a later stage without knowing it at the time it performs the action. 
Additionally, the RL agent needs to use trial-and-error search in order to get a sense of the reward distribution of the different actions. 
Initially the agent doesn’t know the value of actions and needs to try different actions at random, and this is called exploration.
Using that experience the agent can start deterministically choosing actions that generate the most rewards in a state, and this is called exploitation. 
A key part of RL is striking a balance between exploration and exploitation.
If the dynamics of the environment is non-stationary the agent needs to continuously explore to ensure that it is taking the optimal actions as it cannot rely on old estimates. 



## Markov Decision Process 
A Markov Decision Process (MDP) is a stochastic control process, and a classical formalization of sequential decision making. 
MPDs are extensions of Markov chains with actions and rewards. 
MDPs follow Markov property of the future state depending only on the current state $P(X_n=x_n | X_{n-1}=x_{n-1},...,X_0=x_0)=P(X_n=x_n | X_{n-1}=x_{n-1})$. 

> Insert figure of agent-environment interaction

A MPD is a 6-tuple $(\mathcal{S}, \mathcal{A}, p, \mathcal{R}, \gamma, s_0)$.
The agent interacts with the environment at discrete time steps $t=0,1,2,3,...$. 
At each time step $t$, the agent recieves the state of the environment $S_t \in \mathcal{S}$. 
Based on that information the agent chooses one of the actions that is possible in the current state $A_t\in A(s)$. 
When an agent is in a state and performs an action, the transition function $p(s', r | s, a) = Pr\{S_t=s', R_t=r | S_{t-1}=s, A_{t-1}=a\}$ specifies which state the agent will end up in next and what reward it will recieve. 
At the following step the agent recieves a numerical reward $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$, and the next state of the environment $S_{t+1}$. 
The discount factor $\gamma \in [0,1]$ determines present value of future rewards. 
The initial state of the MDP is $s_0$. 
This interaction with a MDP produces a sequence known as a trajectory: $S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, R_3, ...$. 
For continuing tasks this sequence is infinite. 
A finite MDP is a MDP with with finite state space $\mathcal{S}, action space \mathcal{A}, and reward space \mathcal{R}$. 


## Policies and Value Functions. 
Most RL algorithms estimate the value of being in certain states or the value of being in a certain states and performing certain actions. 
A state's value is not only based on the reward an agent recieves while in it. It must also take into account the series of future discounted returns. 
$G_t$ is defined as the future discounted return
\begin{equation}
G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1} = R_{t+1}+ \gamma G_{t+1} 
\end{equation}

A RL agent's objective is to maximize $G_t$. 
It is apparent that $G_t$ will be determined by the actions the agent takes. 
To maximize $G_t$, the agent creates and optimizes a policy. 
A policy $\pi : \mathcal{S} \rightarrow \Delta (\mathcal{A})$ is a mapping from states to a probability distribution over the action space. It is a probability measure $\pi (a | s)$, which is the probability the agent performs action $A_t = a$, given that the current state is $S_t = s$. 
Reinforcement learning algorithms determine how policies are adjusted through experience.

The value function $v_\pi(s)$ is the expected return when starting in state $s$ and following policy $\pi$. It is defined $\forall s \in \mathcal{S}$ as 
\begin{equation}
v_\pi(s)= \mathbb{E}_\pi[G_t | S_t = s] = \sum_{a \in \mathcal{A}(s)} \pi(s|a) \cdot q_{\pi}(s,a)
\end{equation}


The action-value function $q_\pi(s,a)$ is the expected return when performing action $a$ in state $s$, and then following the policy $\pi$. It is defined $\forall s \in \mathcal{S}, a \in \mathcal{A}(s)$ as 
\begin{equation}
q_\pi(s,a)= \mathbb{E}_\pi[G_t | S_t = s, A_t = a] 
\end{equation}





## Bellman Equations
Value and action-value functions can be calculated using the Bellman Equations. 
These equations define a recursive relationship between immediate rewards and the discounted future rewards. 
Inserting the result from (1) into the value function (2) yields
\begin{equation}
v_\pi(s) = \mathbb{E}_\pi [R_{t+1}+\gamma G_{t+1} | S_t = s]
\end{equation}
The value of a state $S_t = s$ when following policy $\pi$ is thus the expected reward $R_{t+1}$ plus the discounted future returns $\gamma G_{t+1}$. 
The expectation for the next reward $R_{t+1}$ can be calculated by summing over all possible action choices weighted by the probability of choosing them according to the policy $\pi$ 
\begin{equation}
v_\pi(s) = \sum_{a\in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}, r \in \mathcal{R}} p(s', r | s, a)[r+\gamma \mathbb{E}_\pi[G_{t+1} | S_{t+1} = s'] ]
\end{equation}
The last expectation is the expected returns from the next state $s'$, i.e., the value function $v_\pi(s')$. 
The Bellman Equation for the value function is defined $\forall s \in \mathcal{S}$ as 
\begin{equation}
v_\pi(s) = \sum_{a\in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}, r \in \mathcal{R}} p(s', r | s, a)[r+\gamma v_\pi (s')] = \sum_{a\in \mathcal{A}} \pi(a|s) \cdot q_\pi(s,a)
\end{equation}

> Figure with bellman equation backup diagram for state value function

The action-value function follows a similar argument. Inserting (1) into the action-value function (3) gives
\begin{equation}
q_\pi(s,a) = \mathbb{E}_\pi [R_{t+1}+\gamma G_{t+1} | S_t = s, A_t = a]
\end{equation}
Unlike for the value function, the next action $A_t = a$ is already known, therefore
\begin{equation}
q_\pi(s,a) = \sum_{s' \in \mathcal{S}, r \in \mathcal{R}} p(s', r | s, a)[r+\gamma \mathbb{E}_\pi[G_{t+1} | S_{t+1} = s'] ]
\end{equation}
The last expectation, which is the value function, can be swapped out for the action-value function using the result from (2). 
Thus, the Bellman Equation can be defined. 
The Bellman Equation for action-value function is defined $\forall s \in \mathcal{S}, a \in \mathcal{A}(s)$ as 
\begin{equation}
q_\pi(s,a) = \sum_{s' \in \mathcal{S}, r \in \mathcal{R}} p(s', r | s, a) \left[r+\gamma \sum_{a' \in \mathcal{A}}\pi(a'|s') \cdot q_\pi(s', a') \right] 
\end{equation}


> Figure with bellman equation backup diagram for action value function



## Optimal Policies and Value Functions. 
The goal of RL is for the agent to learn an optimal or near-optimal policy that maximizes the reward it recieves. 
An optimal policy is denoted by $\pi_*$. There may be more than one. 
The optimal value function $v_*$ is defined $\forall s \in \mathcal{S}$ as 
\begin{equation}
v_*(s) = \max_{\pi} v_\pi(s)
\end{equation}
The optimal action-value function $q_*$ is defined $\forall s \in \mathcal{S}, a \in \mathcal{A}(s)$ as 
\begin{equation}
q_*(s,a) = \max_{\pi} q_\pi(s,a)
\end{equation}


To satisfy the recursive relationship condition for the Bellman equations, the optimal value function $v_*$, being a special case, must be written in a special form without reference to a specific policy. 
The Bellman optimality equation state that the value of a state when following an optimal policy is equal to the action-value of taking the optimal action in that state. 
The Bellman optimality equations for the optimal value function $v_*$ is 
\begin{align*}
	v_*(s)  &= \max_{a \in \mathcal{A}(s)} q_\pi_* (s,a) && \\
	            &= \max_{a \in \mathcal{A}(s)} \mathds{E}_\pi_* [G_t | S_t = s, A_t = a] && \\
	            &= \max_{a \in \mathcal{A}(s)} \mathds{E}_\pi_* [R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a] && \\
	            &= \max_{a \in \mathcal{A}(s)} \mathds{E} [R_{t+1} + \gamma v_*(S_{t+1}) | S_t = s, A_t = a] && \\
	            &= \sum_{s' \in \mathcal{S}, r \in \mathcal{R}} p(s', r | s, a) [r+\gamma v_*(s')]  && \\
\end{align*}

The Bellman optimality equations for the optimal action-value function $q_*$ is 
\begin{align*}
	q_*(s,a)  &= \mathds{E} [R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') | S_t = s, A_t = a] && \\
	            &= \sum_{s' \in \mathcal{S}, r \in \mathcal{R}} p(s', r | s, a) [r+\gamma \max_{a'} q_*(s', a')]  && \\
\end{align*}


## Dynamic Programming.
Dynamic Programming (DP) is a RL algorithm for iteratively evaluate and improve a policy using Bellman equations as updates. 
It assumes perfect knowledge of the MDP, i.e., a finite MDP with where the transition function is known.
DP uses value functions to organize and structure the search for good policies. 
Once the optimal value function (or action-value function) is found, obtaining the optimal policy is trivial. 

The dynamic programming approach can be divided into two steps: policy evaluation and policy improvement. 
Policy evaluation consists of computing the value function  for some policy $\pi$ using the Bellman equation. 
Policy improvement consists of modifying the old policy $\pi$ to greedily choose the optimal action $a$ in state $s$ and otherwise follow the old policy $\pi$. 

The massive assumption of perfect knowledge of the MDP, combined with being very computationally expensive, makes DP not very usable in real world applications. 
Nevertheless, they are theoretically important, and most RL algorithms build on them. 



## Monte Carlo Methods.
In most real world cases the dynamics of the environment is not known, i.e., the transition function $p(s', r | s, a)$ is not known. 
Monte Carlo Methods (MCM) is a way of using simulated experience with the environment to learn optimal behaviour, without knowing the environments dynamics and being able to compute the value function directly like in dynamic programming. 
The approach is based on averaging sample returns after episodes, so it does not bootstrap, and therefore does not introduce bias. 
However, it can not update in a step-by-step (online) sense, only works for episodic tasks. 

MCM follows the same pattern as dynamic programming of policy evaluation and policy improvement. 
The policy evaluation step is based on the idea that the average returns observed after visits to a state will converge to the true value. 
The policy improvement step is the same as for dynamic programming. 
First-visit MCM is when the value function is estimated by averaging the return following the first visit to the state $s$. 
First-visit MCM is when the value function is estimated by averaging the return following every visit to the state $s$. 

A problem with Monte Carlo methods is that certain state-action pairs may not be visited, which does not work for estimation. 
Exploring starts is a way to resolve this issue by giving every state-action pair a non-zero probability of being selected as the start. Every state-action pair will then be visited infinitely often for infinitely many episodes. 


Monte Carlo methods can be used both on- and off-policy.
On-policy is when the same policy that is being evaluated and improved is also used to generate trajectories. 
Off-policy is when a different policy is used to generate the trajetories. 




## Temporal Difference Learning.
Temporal-difference (TD) learning is combines dynamic programming and Monte Carlo methods to make a learning algorithm that learns from experience and does not require knowledge of the dynamics of the environment, and is online. 
The general idea of TD learning is to adjust the Q-values and policy after every observation, and not only after every episode like MCM. 
Since TD learning is fully online it can be used in continous tasks. 

The TD prediction for the value function is updated every step. 
The update is given as $V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$, where $\alpha \in (0,1]$ is the learning rate. 
The TD error $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the estimation error made at step $t$. 
The update is bootstrapping, and therefore introduces bias. 


SARSA is an on-policy TD learning algorithm that immediately update the value of a state-action pair after observing the next reward and state action pair $S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}$
\begin{equation}
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
\end{equation}

Q-learning is an off-policy TD learning algorihm that approximates $q_*$ independent of the policy we follow 
\begin{equation}
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
\end{equation}





## Policy Gradient Methods.
> See policy_gradient_methods.md
