# Policy Gradient Methods.
While value-based RL approaches extract a policy from action-value estimates, Policy Gradient (PG) methods learn a parameterized policy and optimizes it directly. 
PG methods can learn stochastic policies, which is advantageous in problems where there is no optimal deterministic policy, e.g., problems with perceptual aliasing. Policy gradient methods use gradient ascent to optimize the policy as the name implies, and can therefore only guarantee convergence to a local optimum. 
They can handle continuous space, which for value-based methods is far too expensive computationally. They are by definition on-policy. \cite{sutton2018reinforcement}. 



The policy’s parameter vector is $\theta\in \mathbb{R}^{d'}$, with the policy defined as: $ \pi(a|s,\theta)=Pr\{A_t=a|S_t=s,\theta_t=\theta\}$. If the action space is discrete and not too large it's common to parameterize with a preference function $h(a,s,\theta)\in\mathbb{R}$ for every state-action pair \cite{sutton2018reinforcement}. Then a probability distribution of the actions is formed using the action preferences, e.g., using the softmax distribution
\begin{equation}
\pi(a|s,\theta)=\dfrac{e^{h(s,a,\theta)}}{\sum_b e^{h(s,b,\theta)}} 
\end{equation}
where we all action probabilities in each state sum to 1 and the action with highest preference value has the highest chance of being chosen, and so on. The preference function can be parameterized any way you'd like, e.g., linear in features like $h(s,a,\theta)=\theta^\top x(s,a)$, where $x(s,a)\in\mathbb{R}^{d'}$ are feature vectors constructed using feature construction \cite{sutton2018reinforcement}. Essentially we want to increase the $h$ value for actions that produce a lot of reward, which increase the probability of that action being chosen and decrease the probability of all the other actions being chosen. 

For continuous action space we learn statistics of the probability of the action space. Creating a parameterization like that can be done using the Gaussian distribution
\begin{equation}
\pi(a|s,\theta)= \dfrac{1}{\sigma(s,\theta)\sqrt{2\pi}}     e^{-\dfrac{(a-\mu(s,\theta))^2}{2\sigma(s,\theta)^2}}
\end{equation}
where $\mu(s,\theta)\in\mathbb{R}$ and $\sigma(s,\theta)\in\mathbb{R^+}$ are parametric function approximations of the mean and standard deviation respectively. 
The mean decides the space where we will favour actions, while the standard deviation decides the degree of exploration. 
Note that this gives a probability density as opposed to the probability distribution of the softmax distribution. 


In order to optimize the policy $\pi_\theta$ determined by $\theta$ we optimize the parameter $\theta$. We define the the performance measure $J$ for the episodic setting as the true value function for the policy $\pi_\theta$
\begin{equation}
J(\theta)= v_{\pi_\theta} (s_0) = \sum_{a\in A} q_\pi (s_0,a) \pi(a|s_0,\theta)
\end{equation}
where $s_0$ is the start state for the episode. $J$ tells us the total expected reward from state $s_0$ to the final state. Using gradient ascent, we want to move $\theta$ towards the direction suggested by the gradient of $J$ to maximize the reward with respect to $\theta$, thus we get the following gradient ascent update: 
\begin{equation}
\theta_{t+1}=\theta_t+\alpha \widehat{\nabla J(\theta_t)}
\end{equation}
where $\alpha$ is the step-size and $\widehat{\nabla J(\theta_t)}$ is a stochastic estimate whose expectation approximates the gradient of $J$ with respect to $\theta$ \cite{sutton2018reinforcement}. 
The policy gradient theorem provides an expression that is proportional to the gradient: 
\begin{equation}
\nabla J(\theta)\propto \sum_s \mu(s) \sum_a q_\pi (s,a) \nabla \pi(a|s,\theta) \footnote{For the full proof see chapter 13.2 in \cite{sutton2018reinforcement}}
\end{equation}
Importantly this expression does not involve the derivative of the on-policy state distribution, allowing us to simulate paths and update the policy parameter at every step of an episode. \cite{sutton1999policy}. 













## REINFORCE/Monte-Carlo Policy Gradients
REINFORCE works by sampling episodes and updating the policy parameter $\theta$ on every step of the episode. As we have seen PG methods update $\theta$ using gradient ascent. The policy gradient theorem gives us an expression that is proportional to the gradient, and now we need some way of sampling whose expectation equals this expression
\cite{sutton2018reinforcement}. 
We observe that the expression for the gradient in the PG theorem is just a sum over the states weighted by how often they are encountered under the target policy $\pi$. Assuming we follow $\pi$ then we will encounter states in these proportions \cite{sutton2018reinforcement}. Using that we derive the following expression: 
\begin{align*}
	\nabla J(\theta) &\propto \sum_s \mu(s) \sum_a q_\pi (s,a) \nabla \pi(a|s,\theta) && \\
	            &= \mathds{E}_\pi \left[ \sum_a q_\pi (S_t,a)\nabla\pi(a|S_t,\theta) \right] \right] && \\
	            &= \mathds{E}_\pi \left[ \sum_a q_\pi (S_t,a)\nabla\pi(a|S_t,\theta) \dfrac{\pi(a|S_t,\theta)}{\pi(a|S_t,\theta)} \right] \right] && \text{adding } \dfrac{\pi(a|S_t,\theta)}{\pi(a|S_t,\theta)} \\
				&= \mathds{E}_\pi \left[q_\pi (S_t,A_t) \dfrac{\nabla\pi(A_t|S_t,\theta)}{\pi(A_t|S_t,\theta)} \right] && \text{replacing } a \text{ by the sample } A_t\sim\pi\\
				&= \mathds{E}_\pi \left[ G_t \dfrac{\nabla\pi(A_t|S_t,\theta)}{\pi(A_t|S_t,\theta)} \right] && \text{because } \mathds{E}_\pi[G_t|S_t,A_t]=q_\pi(S_t,A_t)\\
\end{align*}
This expression can be sampled on each time step t and its expectation is proportional to the gradient \cite{sutton2018reinforcement}. Now we have all we need to define the policy parameter update for REINFORCE:
\begin{equation}
\theta_{t+1}=\theta_t+\alpha\gamma^t G_t  \dfrac{\nabla\pi(A_t |S_t,\theta_t )}{\pi(A_t |S_t,\theta_t )} 
\end{equation}
where $\alpha$ is the step-size, $\gamma$ is the discount-factor and $G_t$ is the total discounted return. 
The direction of the gradient is in the parameter space that increases the probability of repeating action $A_t$ on visits to $S_t$ in the future the most \cite{sutton2018reinforcement}. The higher the total discounted return, the more we want to repeat that action. The update is inversely proportional to the action probability to adjust for different frequencies of visits to states, i.e., some states might be visited often and have an advantage over less visited states. 
The state-value is equal to the expectation of the estimated final discounted return $G_t$ that is calculated after the episode has terminated. Therefore REINFORCE is a Monte-Carlo (MC) method and unbiased. 
A common problem for MC methods is that there might be large variability of rewards since the trajectory space is large, leading to high variance. 
High variance leads to unstable learning updates and slower convergence. 



## REINFORCE with Baseline
REINFORCE with baseline is a generalization of REINFORCE that attempts to reduce variance without introducing bias. The algorithm works similarly to \ref{sec:REINFORCE}}, but at every step of the episode a baseline is subtracted from the final discounted return $G_t$. 
The baseline can be any random variable or function that does not depend on a. However, it should depend on what state it’s in and be lower for states with low reward actions and higher for states with high reward actions in order to reduce the variability of returns \cite{sutton2018reinforcement}.
A prevalent choice of baseline is $\hat{v}(S_t,w)$ which is an estimate for the state value. The weights vector $w\in \mathbb{R}^d$ is updated along with $\theta$ by $w_{t+1}=w_t+\alpha^w (G_t-\hat{v}(S_t,w_t ))\nabla\hat{v}(S_t,w_t )$. It turns out the PG theorem is still valid when subtracting a baseline: 
\begin{equation}
\nabla J(\theta)\propto \sum_s \mu(s) \sum_a (q_\pi (s,a)-b(s)) \nabla \pi(a|s,\theta) 
\end{equation}
because the subtracted quantity is zero $\sum_a b(s)\nabla\pi(a|s,\theta)=b(s)\nabla\sum_a \pi(a|s,\theta)=b(s)\nabla 1=0$. Therefore by the arguments above and from \ref{sec:REINFORCE} we get the following policy parameter update for REINFORCE with baseline:
\begin{equation}
\theta_{t+1}=\theta_t+\alpha^\theta \gamma^t (G_t-b(S_t ))  \dfrac{\nabla\pi(A_t |S_t,\theta_t )}{\pi(A_t |S_t,\theta_t )} 
\end{equation}
where $\alpha^\theta$ is a different step-size from the weights update $\alpha^w$. Note that the baseline uses the final discounted return after the episode has terminated and not an online estimation, thereby remaining unbiased. 
