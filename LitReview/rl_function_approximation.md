# Function Approximation.
In many real-world applications, the state space is too large to store. 
Not only does it take a lot of memory to store all state values, but computing them is also extremely expensive. 
In these cases, tabular methods are simply not computationally feasible. 
Feature vectors are used instead to represent states when the state space is too large to enumerate. 
If one or more features are real numbers the number of states is infinite. 
In certain problems visiting the exact same state twice is unlikely, and most states will never be visited at all. 
It is therefore necessary to generalize from previous encounters to states with similar characteristics. 
This can be accomplished using function approximation, which is commonly used in many branches of applied mathematics. 
Function approximation takes examples from the desired function, e.g., value function, and attempts to generalize it to approximate the entire function. 
Supervised learning is a form of function approximation. 


## On-policy Prediction with Approximation. 
The on-policy prediction step with approximation is estimating the state-value function $v_\pi(s)$ from on-policy data. 
Instead of a tabular representation, a parameterized function $\hat{v} (s,w)\approx v_\pi(s)$ where $w\in\mathbb{R}^d$ is the weight vector, is used to approximate the value function. 
The approximation $\hat{v}$ may be the function computed by an artificial neural network, where the weight vector $w$ is the connection weights in all the layers. 
Artificial Neural Networks (ANNs) are widely used for nonlinear function approximation \cite{sutton2018reinforcement}. 
The advantage of nonlinear function approximation is that it can automatically create appropriate features through training. 
The result is hierarchical representations that do not require manual feature engineering. 
Typically, $\dim{w} \ll |\mathcal{S}|$.
By updating one weight, many states' estimated values will be affected, so the 
change from updating one state will generalize to affect many others. 
In reinforcement learning the target function may be nonstationary, as opposed to conventional supervised learning function approximation. 


In the tabular setting, the estimated state values can converge to the true value function. Keeping a continuous prediction quality measure is therefore not necessary. 
For function approximation, however, we need some way of measuring performance, as we cannot update the policy in a way that ensures improvement. 
The number of states is assumed to be much greater than the number of weights. 
Updating the weights to improve the accuracy of one state estimate will therefore make others' less accurate. 
The state distribution $\mu(s)\geq 0$, $\sum_{s\in \mathcal{S}} \mu(s)=1$ is used to specify how much we care about the estimates of each state. 
The state distribution $\mu(s)$ is often defined to represent the fraction of time spent in $s$. For on-policy algorithms, this is known as the on-policy state distribution. 
In the continuing setting, this is the stationary distribution under $\pi$. 
The objective function for on-policy prediction is the Means Squared Value Error $\overline{VE}$
\begin{equation}
\overline{VE} (w) = \sum_{s\in\mathcal{S}}\mu(s)[v_\pi (s)- \hat{v}(s,w)]^2
\end{equation}

Ideally, the the estimate would converge to a global optima, i.e., we would find weights $w^*$ such that $\overline{VE}(w*) \leq \overline{VE}(w)$, $\forall w$. 
Unfortunately, there are no convergence guarantees for prediction with function approximation. 


### Optimization. 
Stochastic Gradient Descent (SGD) is a method for iteratively optimizing the weights of an objective function. SGD is well suited for online reinforcement learning. 
Stochastic gradient descent replaces the gradient in conventional gradient descent with a stochastic approximation. The stochastic approximation is only calculated on a subset of the data. This reduces the computational costs of high-dimensional optimization problems. 
The gradient of the value function can be computed at every step $t=0,1,2,3,...$, and the weights $w_t$ adjusted in the direction suggested by the gradient $\nabla \hat{v}(S_t,w_t)$, in order to minimize $\overline{VE}$. 
Thus, the approximate value function $\hat{v} (s,w)$ needs to be differentiable with respect to $w$. 
State distributions $\mu$ are assumed to be the same in the examples under which $\overline{VE}$ is minimized. 

For the update in stochastic gradient methods we define $U$ as an unbiased approximation of $v_\pi(S_t))$, i.e., $\mathbb{E}[U_t | S_t =s]=v_\pi(S_t)$. 
$U$ can be the Monte Carlo target $U_t=G_t$, since it is an unbiased estimate of $v_\pi(S_t$. 
For every step $t$, the weights $w_t$ are adjusted in the direction that most reduces the error to the new value. The update is defined as
\begin{equation}
w_{t+1} = w_t + \alpha [U_t(S_t) - \hat{v}(S_t, w_t)] \nabla \hat{v}(S_t, w_t)
\end{equation}
where $\alpha\in[0,1]$ is the step-size. Stochastic approximation theory guarantees convergence to a local optima if $\alpha$ satisfies the conditions $\sum \alpha = \infty$ and $\sum \alpha^2 < \infty$. 

Convergence is not guaranteed when using bootstrapping estimates of $v_\pi(S_t)$. 
Stochastic semi-gradient methods are based on bootstrapping estimates. 
If bootstrapping estimates of $v_\pi(S_t))$ are used, the target depends on the current weight vector. 
They take changes to $w$ into account for the value estimates and ignore the effect of changes to $w$ on the gradient estimate. They are therefore biased and not true gradient descent methods as they only take part of the gradient into account. 
For the stochastic semi-gradient update we define $U$ as a biased approximation of $v_\pi(S_t))$. 
Convergence is not robust, but stochastic semi-gradient methods are often preferred as learning is faster and online. 








## On-policy Control with Approximation. 
The goal of the control problem in RL is to find the optimal policy. 
Measuring the performance of policies for problems where there are no clearly identified states, e.g., states that are represented by feature vectors, presents a challenge. If it is difficult to differentiate between the states, then the trajectories in practice only consist of rewards and actions. 
Thus, performance must be assessed from rewards and actions alone. 
One strategy for doing that is known as the average reward setting. 
Average reward RL is an optimization problem. 
For the continuous (undiscounted) case the average rate of reward while following the policy $\pi$ is used as a quality measure for policies 
\begin{align*}
 r(\pi) &= \lim_{h \rightarrow \infty} \dfrac{1}{h} \sum_{t=1}^h \mathbb{E}[R_t | S_0, A_{0:t-1} \sim \pi] && \\
 &= \lim_{t \rightarrow \infty} \mathbb{E}[R_t | S_0, A_{0:t-1} \sim \pi] && \\
 &= \sum_{s} \mu(s) \sum_{a} \pi(a|s) \sum_{s',r} p(s', r | s, a)r && \\
\end{align*}
where $\mu_\pi(s)$ is the steady-state distribution under $\pi$, defined as $\mu_\pi(s) = \lim_{t \rightarrow \infty} Pr\{S_t=s | A_{0:t} \sim \pi \}$, $\forall \pi$, that is assumed to be independent of $S_0$. 
The system is assumed to be ergodic, i.e., the expectations of being in a state depend only on the policy and MDP transition probabilities, and not on early choices. 


Differential returns are used in the average reward setting and defined as $G_t = \sum_{k=t+1} R_{k} - r(\pi)$. 
The differential value functions in the continuous setting are defined like before. 
The differential Bellman equations are modified by removing the discount factor $\gamma$ and differential returns are inserted instead of normal (discounted) returns. Thus, the differential Bellman equations are defined as
\begin{equation}
v_\pi (s) = \sum_{a\in\mathcal{A}(s)} \pi(a|s) \sum_{r\in\mathcal{R}, s'\in\mathcal{S}} p(s', r | s, a) \left[r-r(\pi)+v_\pi(s') \right]
\end{equation}
\begin{equation}
q_\pi(s,a)=\sum_{r\in\mathcal{R}, s'\in\mathcal{S}} \left[ r - r(\pi) + \sum_{a' \in\mathcal{A}(s')} \pi(a'|s')q_\pi(s',a') \right]
\end{equation}
\begin{equation}
v_* (s) = \max_{a\in\mathcal{A}(s)} \sum_{r\in\mathcal{R}, s'\in\mathcal{S}} p(s', r | s, a) \left[r- \max_\pi{r(\pi)}+v_*(s') \right]
\end{equation}
\begin{equation}
q_*(s,a)=\sum_{r\in\mathcal{R}, s'\in\mathcal{S}} \left[ r - \max_\pi{r(\pi)} + \max_{a' \in\mathcal{A}(s')} q_*(s',a') \right]
\end{equation}



Discounting is applied to tasks where immediate rewards are valued higher than future rewards. 
It has the additional benefit of bounding the return. 
If $\gamma < 1$, then $\sum_k \gamma^k=\dfrac{1}{1-\gamma}$, and $G_t$ is bounded, as long as $R_t$ is also bounded. 
In the continuous setting, the discounted return $G_t$ could be measured at every time step $t$. 
However, it turns out that the discounted average returns are proportional to the un-discounted average returns \cite{sutton2018reinforcement}. The discount factor does not change the order of the policies, so it does not affect the problem \footnote{See the box on page 254 in \cite{sutton2018reinforcement} for proof.}. 
Accordingly, discounting is inconsequential when defining the control problem with function approximation.
Discounted RL is incompatible with function approximation for control in continuing tasks, as it is not an optimization problem \cite{naik2019discounted}. 
In the discounted setting there is no objective function that can describe the quality of every policy.
Providing a partial order between policies is unfeasible for massive or continuous action and state spaces. 
Even using a compact policy representation is unfeasible, as there is usually not one representable policy that is unambiguously better than every other representable policy. 
A policy might have a higher value in some states but lower in others. 
Therefore, an objective function, like the total sum of rewards in the episodic setting, is necessary to rank the quality of policies in the continuing setting.
Average reward is an optimization problem that is well suited for continuous-time RL. 
It turns out that the discounted average returns are proportional to the un-discounted average returns \cite{sutton2018reinforcement}, and the discount factor does not change the partial order of the policies \footnote{See the box on page 254 in \cite{sutton2018reinforcement} for proof.}. 
Therefore, the discount factor becomes a hyperparameter of the algorithm, not a parameter specifying the optimization objective \cite{sutton2018reinforcement}. 
Even more, the discount factor can change the optimization objective \cite{naik2019discounted}. 
Changing the discount factor will sometimes change the policy when using algorithms like SARSA or Q-learning. Greedily maximizing discounted future value does therefore not maximize average reward. 




## Eligibility Traces.
One of the distinguishing features of reinforcement learning is delayed reward. 
Updating state values based on future rewards the agent has yet to observe is a fundamental challenge of RL. 
Temporal difference and Monte Carlo methods solve this problem differently. 
Monte Carlo methods wait for the episode to finish, while one-step TD-learning looks to the next state. 
$n$-step TD-learning is a middle ground that looks into the next $n$ states. 
Eligibility traces used together with TD errors offer an elegant algorithmic mechanism of unifying TD and Monte Carlo methods. 
Eligibility traces are more computationally efficient than normal $n$-step methods as it only stores a single trace vector instead of a list of the previous $n$ feature vectors. 
Learning is continuous and uniformly in time rather than episodic. Furthermore, learning can affect behavior instantly after visiting a state rather than $n$ steps later. 

The eligibility trace is a memory vector $z_t \in \mathbb{R}^d$ where $d$ is the dimension of weight vector $w_t$. 
The eligibility trace $z_t$ keeps track of each component of the weights $w_t$ contribution to recent state values, decaying over time. 
Those components of $w_t$ will learn if a nonzero TD error occurs before the trace falls back to zero. 
The trace decay parameter $\lambda \in [0,1]$ determines the rate at which the trace falls. 
$\lambda = 0$ yields one-step TD methods, $\lambda = 1$ yields Monte Carlo methods, and $\lambda \in (0,1)$ yields some intermediate method. 
For $\lambda \in [0,1]$ the weights $(1-\lambda) = \sum_{n=1}^\infty \lambda^{n-1} = 1$. 
Eligibility traces are initialized to zero $z_{-1}=0$, and updated as follows
\begin{equation}
z_t = \gamma \lambda z_{t-1} + \nabla \hat{v}(S_t, w_t)
\end{equation}
They are then used as a scaling factor for the TD error when updating state value approximations. 
