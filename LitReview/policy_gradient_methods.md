# Policy Gradient Methods.
While value-based RL approaches extract a policy from action-value estimates, Policy Gradient (PG) methods learn a parameterized policy and optimize it directly. 
PG methods can learn stochastic policies, which is advantageous in problems where there is no optimal deterministic policy, e.g., problems with perceptual aliasing. Policy gradient methods use gradient ascent to optimize the policy as the name implies, and can therefore only guarantee convergence to a local optimum. 
They can handle continuous space, which for value-based methods is far too expensive computationally. They are by definition on-policy. \cite{sutton2018reinforcement}. 



The policy’s parameter vector is $\theta\in \mathbb{R}^{d'}$, with the policy defined as: $ \pi(a|s,\theta)=Pr\{A_t=a|S_t=s,\theta_t=\theta\}$. If the action space is discrete and not too large it's common to parameterize with a preference function $h(a,s,\theta)\in\mathbb{R}$ for every state-action pair \cite{sutton2018reinforcement}. Then a probability distribution of the actions is formed using the action preferences, e.g., using the softmax distribution
\begin{equation}
\pi(a|s,\theta)=\dfrac{e^{h(s,a,\theta)}}{\sum_b e^{h(s,b,\theta)}} 
\end{equation}
where all action probabilities in each state sum to 1 and the action with the highest preference value has the highest chance of being chosen, and so on. 
The preference function can be parameterized by an artificial neural network, where the policy parameters $\theta$ are the connection weights of the neural net \cite{sutton2018reinforcement}. 
Increasing $h$ for actions that produce a lot of rewards increases the probability of choosing those actions and decreases $h$ for all other actions, which in turn decreases the probability of choosing one of those actions.

Continuous action space is modeled by learning the statistics of the probability of the action space. 
Creating a parameterization like that can be done using the Gaussian distribution
\begin{equation}
\pi(a|s,\theta)= \dfrac{1}{\sigma(s,\theta)\sqrt{2\pi}} e^{-\dfrac{(a-\mu(s,\theta))^2}{2\sigma(s,\theta)^2}}
\end{equation}
where $\mu(s,\theta)\in\mathbb{R}$ and $\sigma(s,\theta)\in\mathbb{R^+}$ are parametric function approximations of the mean and standard deviation respectively. 
The policy parameter vector is divided into two parts, $\theta = [\theta_\mu, \theta_\sigma]^\top$. 
The mean decides the space where the agent will favor actions, while the standard deviation decides the degree of exploration. 
The mean can be approximated as a linear function $\mu(s,\theta) = {\theta_\mu}^\top x_\mu(s)$, and the standard deviation as the exponential of a linear function $\sigma(s,\theta)=e^{{\theta_\sigma}^\top x_\sigma(s)}$, where $x_\mu(s)$ and $x_\sigma(s)$ are state feature vectors. 
It is important to note that this gives a probability density, not a probability distribution like the softmax distribution. 


In order to optimize the policy $\pi_\theta$ its parameter $\theta$ is optimized. 
The performance measure $J$ for the episodic setting is defined as the true value function for the policy $\pi_\theta$
\begin{equation}
J(\theta)= v_{\pi_\theta} (s_0) = \sum_{a\in A} q_\pi (s_0,a) \pi(a|s_0,\theta)
\end{equation}
where $s_0$ is the start state for the episode. $J$ expresses the total expected reward from state $s_0$ to the final state. 
The policy parameter $\theta$ is moved in the direction suggested by the gradient of $J$ to maximize the reward with respect to $\theta$, yielding the following gradient ascent update
\begin{equation}
\theta_{t+1}=\theta_t+\alpha \widehat{\nabla J(\theta_t)}
\end{equation}
where $\alpha$ is the step-size and $\widehat{\nabla J(\theta_t)}$ is a stochastic estimate whose expectation approximates the gradient of $J$ with respect to $\theta$ \cite{sutton2018reinforcement}. 
The policy gradient theorem provides an expression that is proportional to the gradient: 
\begin{equation}
\nabla J(\theta)\propto \sum_s \mu(s) \sum_a q_\pi (s,a) \nabla \pi(a|s,\theta) \footnote{For the full proof see chapter 13.2 in \cite{sutton2018reinforcement}}
\end{equation}
Importantly this expression does not involve the derivative of the on-policy state distribution, allowing the agent to simulate paths and update the policy parameter at every step of an episode. \cite{sutton1999policy}. 













## REINFORCE/Monte-Carlo Policy Gradients
REINFORCE works by sampling episodes and updating the policy parameter $\theta$ on every step of the episode. Policy gradient methods update $\theta$ using gradient ascent. The policy gradient theorem provides an expression that is proportional to the gradient, and all that is needed some way of sampling whose expectation equals or approximates this expression \cite{sutton2018reinforcement}. 
The expression for the gradient in the PG theorem is just a sum over the states weighted by how often they are encountered under the target policy $\pi$. Assuming the agent follow the policy $\pi$ then it will encounter states in these proportions \cite{sutton2018reinforcement}. This leads to the following expression
\begin{align*}
 \nabla J(\theta) &\propto \sum_s \mu(s) \sum_a q_\pi (s,a) \nabla \pi(a|s,\theta) && \\
 &= \mathds{E}_\pi \left[ \sum_a q_\pi (S_t,a)\nabla\pi(a|S_t,\theta) \right] \right] && \\
 &= \mathds{E}_\pi \left[ \sum_a q_\pi (S_t,a)\nabla\pi(a|S_t,\theta) \dfrac{\pi(a|S_t,\theta)}{\pi(a|S_t,\theta)} \right] \right] && \text{adding } \dfrac{\pi(a|S_t,\theta)}{\pi(a|S_t,\theta)} \\
 &= \mathds{E}_\pi \left[q_\pi (S_t,A_t) \dfrac{\nabla\pi(A_t|S_t,\theta)}{\pi(A_t|S_t,\theta)} \right] && \text{replacing } a \text{ by the sample } A_t\sim\pi\\
 &= \mathds{E}_\pi \left[ G_t \dfrac{\nabla\pi(A_t|S_t,\theta)}{\pi(A_t|S_t,\theta)} \right] && \text{because } \mathds{E}_\pi[G_t|S_t,A_t]=q_\pi(S_t,A_t)\\
 &= \mathds{E}_\pi \left[ G_t \nabla \log{\pi(A_t|S_t,\theta)} \right] && \\
\end{align*}
This expression can be sampled on each time step t and its expectation is proportional to the gradient \cite{sutton2018reinforcement}. 
With everything in place, the policy parameter update for REINFORCE can be defined as
\begin{equation}
\theta_{t+1}=\theta_t+\alpha\gamma^t G_t \dfrac{\nabla\pi(A_t |S_t,\theta_t )}{\pi(A_t |S_t,\theta_t )} 
\end{equation}
where $\alpha$ is the step size, $\gamma$ is the discount factor and $G_t$ is the total discounted return. 
The direction of the gradient is in the parameter space that increases the probability of repeating action $A_t$ on visits to $S_t$ in the future the most \cite{sutton2018reinforcement}. The higher the total discounted return, the more the agent want to repeat that action. The update is inversely proportional to the action probability to adjust for different frequencies of visits to states, i.e., some states might be visited often and have an advantage over less visited states. 
The state-value is equal to the expectation of the estimated final discounted return $G_t$ that is calculated after the episode has terminated. Therefore REINFORCE is a Monte-Carlo (MC) method and unbiased. 
A common problem for MC methods is that there might be large variability of rewards since the trajectory space is large, leading to high variance. 
High variance leads to unstable learning updates and slower convergence. 



## REINFORCE with Baseline
REINFORCE with baseline is a generalization of REINFORCE that attempts to reduce variance without introducing bias. The algorithm works similarly to \ref{sec:REINFORCE}}, but at every step of the episode, a baseline is subtracted from the final discounted return $G_t$. 
The baseline can be any random variable or function that does not depend on a. However, it should depend on what state it’s in and be lower for states with low reward actions and higher for states with high reward actions in order to reduce the variability of returns \cite{sutton2018reinforcement}.
A prevalent choice of baseline is $\hat{v}(S_t,w)$ which is an estimate for the state value. The weights vector $w\in \mathbb{R}^d$ is updated along with $\theta$ by $w_{t+1}=w_t+\alpha^w (G_t-\hat{v}(S_t,w_t ))\nabla\hat{v}(S_t,w_t )$. It turns out the PG theorem is still valid when subtracting a baseline: 
\begin{equation}
\nabla J(\theta)\propto \sum_s \mu(s) \sum_a (q_\pi (s,a)-b(s)) \nabla \pi(a|s,\theta) 
\end{equation}
because the subtracted quantity is zero $\sum_a b(s)\nabla\pi(a|s,\theta)=b(s)\nabla\sum_a \pi(a|s,\theta)=b(s)\nabla 1=0$. Therefore, the arguments above and from \ref{sec:REINFORCE} yields the following policy parameter update for REINFORCE with baseline 
\begin{equation}
\theta_{t+1}=\theta_t+\alpha^\theta \gamma^t (G_t-b(S_t )) \dfrac{\nabla\pi(A_t |S_t,\theta_t )}{\pi(A_t |S_t,\theta_t )} 
\end{equation}
where $\alpha^\theta$ is a different step-size from the weights update $\alpha^w$. Note that the baseline uses the final discounted return after the episode has terminated and not an online estimation, thereby remaining unbiased. 



## Actor-Critic
Policy-based reinforcement learning is better for online learning and stochastic environments, while value-based RL is more sample efficient. 
Actor-Critic (AC) methods seek to combine policy-based methods using parameterized policies with value-based methods using value function approximation. 
The policy-based actor chooses actions, and the value-based critic critique those actions. 
The critic approximates a value function, e.g., a sate-value function $\hat{v}(S_t, w)$ with weight vector $w\in\mathbb{R}^d$, and calculates the TD error $\delta_t = G_{t:t+1} - \hat{v}(S_t, w) = R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)$ after each action. 
Positive errors suggest that the action should be performed more frequently, and negative errors suggest that it should be performed less frequently. 
The actor learns a parameterized policy $\pi(a|s,\theta)$, and optimizes the policy parameters $\theta$ at every time step $t$ using stochastic gradient descent in the direction suggested by the critic. 
The actor's policy is independent of the critic's value function. 
The critic's value function is also optimized using stochastic gradient methods where the gradient is given by $\nabla w = \beta \delta_t \nabla_w \hat{v}(S_t, w) $, where $\beta$ is the learning rate. 
Actor-critic methods are always on-policy as the critic can only critique the actor's policy, which is the policy that generates the trajectories. 
Using the TD error means that actor-critic methods are fully online, and it reduces variance which leads to faster convergence. This introduces bias, however, since it is bootstrapping. 
REINFORCE with baseline is not an AC method since the state-value function is used as a baseline and not as a critic. The state-value is updated through Monte Carlo updates, so it is not bootstrapping. 


> Figure showing actor-critic architecture


Actor-Critic can handle the online and continuous time setting. 
As states, actions, and rewards are encountered, they are processed and never revisited. 
In the continuous setting performance is defined as average reward per time step $J{\theta} = r(\pi)$. 
The pseudocode for actor-critic in the continuous time setting is given below

\begin{algorithm}
\caption{Actor-Critic for Continuous Time Setting}\label{alg:cap}
Input: a differentiable policy parameterization $\pi(a|s, \theta)$ 

Input: a differentiable state-value function parameterization $\hat{v}(s, w)$

Algorithm parameters: $\lambda^w\in[0,1]$, $\lambda^\theta\in[0,1]$, $\alpha^w>0$, $\alpha^\theta>0$, $\alpha^{\bar{R}}>0$

Initialize $\bar{R}\in\mathbb{R}$ (e.g., to $0$)

Initialize state-value weights $w\in\mathbb{R}^d$ and policy parameter $\theta\in\mathbb{R}^{d'}$ (e.g., to $0$)

Initialize $S\in\mathcal{S}$ (e.g., to $s_0$)
\begin{algorithmic}
\State $z^w \gets 0$ (d-component eligibility trace vector)
\State $z^\theta \gets 0$ (d'-component eligibility trace vector)
\While{True}
 \State $A \sim \pi(\cdot |S, \theta)$
 \State Take action $A$, observe $S', R$
 \State $\delta \gets R - \bar{R} + \hat{v}(S', w) - \hat{v}(S, w)$
 \State $\bar{R} \gets \bar{R} + \alpha^{\bar{R}}\delta$
 \State $z^w \gets \lambda^w z^w + \nabla \hat{v}(S,w) $
 \State $z^\theta \gets \lambda^\theta z^\theta + \nabla \ln{\pi(A|S,\theta)} $
 \State $w \gets w + \alpha^w \delta z^w$
 \State $\theta \gets \theta + \alpha^\theta \delta z^\theta$
 \State $S \gets S'$
\EndWhile
\end{algorithmic}
\end{algorithm}



Advantage Actor-Critic (A2C) makes the critic learn the advantage function $A(S_t,A_t)=q(S_t,A_t)-v(S_t)=R_t + \gamma v(s') - v(s)$ instead of the value function. The advantage function expresses how much better an action is compared to the average value of that state. The TD error is a good estimator of the advantage function. 
Optimizing with respect to the advantage function reduces the high variance of policy networks and stabilizes the model. 

Asynchronous Advantage Actor-Critic (A3C) utilizes multiple independent agent networks with their own unique weight vectors that interact with a different copy of the same environment in parallel. 





## Deterministic Policy Gradient. 
Policy gradient methods traditionally sample actions stochastically. 
Deterministic Policy Gradient (DPG) implements policy gradient methods using a deterministic policy. 
It was introduced by researchers from DeepMind in 2014 \cite{silver2014deterministic}. 
The deterministic policy gradient theorem is a special case of the policy gradient theorem where the variance (i.e., exploration) approaches zero \footnote{For full proof see \cite{silver2014deterministic}}. 
One of the advantages of DPG is that the deterministic policy gradient can be estimated more efficiently than the stochastic policy gradient. 
In the stochastic case, the gradient integrates over the whole state and action space, while in the deterministic case the gradient only integrates over the state space. 
In general, deterministic policies do not ensure sufficient exploration, which can lead to premature convergence. 
DPG solves this problem by introducing an off-policy actor-critic algorithm that learns a deterministic target policy from an exploratory behavior policy. 
DPG can outperform stochastic policy gradient methods, especially in high-dimensional action spaces. 




## Trust Region Policy Optimization. 
Policy gradient methods use gradient descent when updating the policy parameter.
The policy parameter weights are moved in the direction suggested by the first-order derivative, i.e., the surface of the gradient is assumed to be flat. 
However, if the surface is curved, a too-big step might hurt the performance of the policy. 
This problem can be mitigated using small step sizes, but too small step sizes lead to slow learning and poor sample efficiency. 
Trust Region Policy Optimization (TRPO) \cite{schulman2015trust} is an algorithm for policy optimization with guaranteed monotonic improvement. 
The theoretical TRPO update is defined as 
\begin{equation}
\theta_{k+1} = \argmax_\theta \mathcal{L}(\theta_k, \theta)
\end{equation}
such that $\bar{D}_{KL} \leq \delta$. The surrogate advantage $\mathcal{L}(\theta_k, \theta)$ measures the performance of the new policy using data from the old policy, and defines the objective function of TRPO
\begin{equation}
J^{TRPO}(\theta) = \mathcal{L}(\theta_k, \theta) = \mathbb{E} \left[ \dfrac{\pi_\theta (a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k} (s,a)} \right]
\end{equation}
TRPO restricts the old and new policies to be within some distance $\delta$, measured by KL-divergence
\begin{equation}
\bar{D}_{KL} (\theta | \theta_k) = \mathbb{E}[D_{KL}(\pi_{\theta}(\cdot | s) || \pi_{\theta_k}(\cdot | s))]\leq \delta
\end{equation}
The theoretical TPRO update can be challenging to work with, so there have been developed more practical implementations that are easier to work with \cite{schulman2015trust}. 
TRPO is not compatible with architectures that includes a lot of noise or parameter sharing. 






## Proximal Policy Optimization. 
Proximal Policy Optimization (PPO) is a first-order optimization and an improvement on TRPO that is more general and simpler to implement. 
It was introduced by researchers from OpenAI in 2017 \cite{schulman2017proximal}
PPO defines a probability ratio between the old and new policy defined as 
\begin{equation}
r(\theta) = \dfrac{\pi_\theta (a|s)}{\pi_{\theta_k} (a|s)}
\end{equation}
Thus, the objective function from TRPO can be expressed as 
\begin{equation}
J^{TRPO}(\theta) = \mathbb{E} \left[ r(\theta) A^{\pi_{\theta_k} (s,a)} \right]
\end{equation}
Instead of adding the KL-divergence constraint from TRPO, PPO restricts the probability ratio $r(\theta)$ to stay withing a small neighborhood around 1 with radius $\epsilon$, which is a hyperparameter (e.g., $\epsilon=0.2$). Thus, the objective function for PPO is 
\begin{equation}
J^{CLIP}(\theta) = \mathcal{L}(\theta_k, \theta) = \mathbb{E} \left[ \min{(r(\theta) A^{\pi_{\theta_k} (s,a)}, clip(r(\theta), 1-\epsilon, 1+\epsilon)A^{\pi_{\theta_k} }) \right]
\end{equation}
where the $clip$ function limits the probability ratio between $[1-\epsilon, 1+\epsilon]$. PPO's objective function $J^{CLIP}$ takes the minimum value between the original value and the clipped value. I.e., the objective is a lower bound. 
