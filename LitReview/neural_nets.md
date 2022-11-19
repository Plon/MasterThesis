# Artificial Neural Networks. 
Supervised learning is a form of function approximation. 
Artificial Neural Networks (ANNs) are the most common supervised learning approach for nonlinear models. 
A neural network is a collection of nodes that is organized in layers as a directed and weighted graph. 
The nodes of an ANN are typically separated into layers; the input layer, one or more hidden layers, and the output layer. Their dimensions depend on the function being approximated. 
The network defines a mapping $h_{\theta} : \mathbb{R}^n \rightarrow \mathbb{R}^m$ where $n$ is the input dimension, $m$ is the output dimension, and $\theta$ is the weights of the network. 



## Feedforward Neural Networks. 
Feedforward neural networks (FNNs) define the class of the simplest neural network where the connections are a directed acyclic graph that only allows signals to travel forward in the network. 
A feedforward network is a mapping $h_\theta$ that is a composition of multivariate functions $f_1, f_2, ...,f_k,$ and $g$, where $k$ is the number of layers in the neural network, defined as $h_\theta = g \circ f_k \circ ... \circ f_2 \circ f_1(x)$. 
The functions $f_j$, $j=1,2,...,k$ represent the hidden layers of the network and are themselves composed multivariate functions. 
The function at layer $j$ is defined as $f_j (x) = a_j(\theta_j x + b_j)$ where $a_j$ is the activation function and $b_j$ is the bias at layer $j$. 
The activation function is used to add nonlinearity to the network. 
The final output layer function $g$ of the network can be tailored to suit the specific problem the network is solving, e.g., linear for Gaussian output distribution or Softmax distribution for Categorical output distribution. 


> Figure showing multi-layer perceptron network. 



## Gradient-Based Learning. 
Neural nets are optimized by adjusting their weights. In the same way as other supervised learning approaches, neural nets are optimized using objective functions. 
Let $J(\theta)$ define the differentiable objective function for a neural network, where $\theta$ are the weights of the network. 
The choice of the objective function and whether it should be minimized or maximized depends on the problem being solved. 
For regression tasks, the objective function is usually minimizing some loss functions like mean-squared error (MSE)
\begin{equation}
J(\theta) = \dfrac{1}{n} \sum_{i=1}^n \left(h_{\theta}(x_{i})-y_{i} \right)^2
\end{equation}


Due to neural nets' nonlinearity, most loss functions are non-convex, meaning that it is not possible to find an analytical solution to $\nabla J(\theta)=0$ as there is for convex optimization algorithms. 
Instead, neural networks are usually optimized using iterative, gradient-based optimization algorithms \cite{goodfellow2016deep}. There are no convergence guarantees. 
Gradient descent-based algorithms adjust the weights $\theta$ in the direction that minimizes the MSE loss function. 
The update rule for parameter weights in gradient descent is defined as
\begin{equation}
\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)
\end{equation}
where $\alpha>0$ is the learning rate and the gradient $\nabla J(\theta_t)$ is the partial derivatives of the objective function with respect to each weight. The learning rate defines the rate at which the weights move in the direction suggested by the gradient of the cost function. 



### Stochastic Gradient Descent. 
Optimization algorithms that process the entire training set simultaneously are known as batch learning algorithms. 
Using the average of the entire training set allows for calculating a more accurate gradient estimate. 
The speed at which batch learning converges to a local minima will therefore be faster than online learning. However, batch learning is not suitable for all problems, e.g., problems with huge datasets due to the high computational costs of calculating the full gradient, or problems with dynamic probability distributions. 


Instead, Stochastic Gradient Descent (SGD) is often used in optimizing neural networks. 
SGD replaces the gradient in conventional gradient descent with a stochastic approximation. The stochastic approximation is only calculated on a subset of the data. This reduces the computational costs of high-dimensional optimization problems. 
However, by using a stochastic estimate of the gradient, the loss is not guaranteed to be strictly decreasing. 
SGD is often used for problems with continuous streams of new observations, rather than a fixed-size training set. 
The update rule for SGD is similar to the one for GD but replaces the true gradient with a stochastic estimate 
\begin{equation}
\theta_{t+1} = \theta_t - \alpha_t \nabla_\theta J^{(j)}(\theta_t)
\end{equation}
where $\nabla_\theta J^{(j)}(\theta)$ is the stochastic estimate of the gradient computed from observation $j$. The total loss is defined as $J(\theta) = \sum_{j=1}^N J^{(j)}(\theta)$ where $N\in\mathbb{N}$ is the total number of observations. 
The learning rate at time $t$ is defined as $\alpha_t>0$. Due to the noise introduced by the SGD gradient estimate, it is crucial to gradually decrease the learning rate over time to ensure convergence. Stochastic approximation theory guarantees convergence to a local optima if $\alpha$ satisfies the conditions $\sum \alpha = \infty$ and $\sum \alpha^2 < \infty$. 
It is common to adjust the learning rate using the following update rule $\alpha_t = (1-\beta)\alpha_0 + \beta \alpha_\tau$, where $\beta = \dfrac{t}{\tau}$, and the iteration is kept constant after $\tau$ iterations \cite{goodfellow2016deep}. 


Due to hardware parallelization, computing the gradient of $N$ observations simultaneously will usually be faster than computing each gradient separately \cite{goodfellow2016deep}. 
Neural networks are therefore often trained on sets of more than one but less than all observations, known as mini-batch learning. 
Mini-batch learning is an intermediate approach to fully online learning and batch learning where weights are updated simultaneously after accumulating gradient information over a subset of the total observations. 
In addition to providing better estimates of the gradient, mini-batches are more computationally efficient than online learning while still allowing training weights to be adjusted periodically during training. 
Therefore, minibatch learning can be used to learn systems with dynamic probability distributions. 
Samples of the mini-batches should be independent and drawn randomly. 
Drawing ordered batches will result in biased estimates, especially for data with high temporal correlation. 



Stochastic gradient descent and mini-batches of small size will exhibit higher variance than conventional gradient descent during training due to noisy gradient estimates. 
The higher variance can be useful to escape local minima and potentially find new, better local minima. 
However, high variance can also lead to problems such as overshooting and oscillation that can cause the model to fail to converge. 
Several extensions have been made to stochastic gradient descent to circumvent these problems. 



#### Adaptive Gradient Algorithm. 
The Adaptive Gradient (AdaGrad) is an extension to stochastic gradient descent introduced in 2011 \cite{duchi2011adaptive}. 
It outlines a strategy for adjusting the learning rate to converge quicker and improving the capability of the optimization algorithm. 
A per-parameter learning rate allows AdaGrad to improve performance on problems with sparse gradients. Learning rates are assigned to be lower for parameters with frequently occurring features, and higher for parameters with less frequently occurring features. 
The AdaGrad update rule is given as 
\begin{equation}
\theta_{t+1} = \theta_t - \dfrac{\alpha}{\sqrt{G_t + \epsilon}} g_t
\end{equation}
where $g_t=\nabla_\theta J^{(j)}(\theta_t)$, and $G_t = \sum_{\tau=1}^t {g_t g_t^\top}$, is the outer product of all previous subgradients. $\epsilon > 0$ is a smoothing term to avoid division by zero. 
As training proceeds, the squared gradients in the denominator of the learning rate will continue to grow, resulting in a strictly decreasing learning rate. At some point, the learning rate will become so small that the model cannot acquire any new information. 




#### Root Mean Square Propagation. 
Root Mean Square Propagation (RMSProp) is an unpublished extension to SGD developed by Geoffrey Hinton. 
RMSProp was developed to resolve the problem of AdaGrad's diminishing learning rate. 
Like AdaGrad, it maintains a per-parameter learning rate. 
To normalize the gradient, it keeps a moving average of squared gradients. This normalization decreases the learning rate for larger gradients to avoid the exploding gradient problem and increases it for smaller gradients to avoid the vanishing gradient problem. 
The RMSProp update rule is given as
\begin{equation}
\theta_{t+1} = \theta_t - \dfrac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} g_t
\end{equation}
where $E[g^2]_t = \beta E[g^2]_t + (1-\beta)g^2_t$ where $v_t$ is the exponentially decaying average of squared gradients and $\beta>0$ is a second learning rate conventionally set to $\beta=0.9$.



#### Adam. 
The Adam optimization algorithm is an extension of stochastic gradient descent that has recently seen wide adoption in deep learning. It was introduced in 2015 \cite{kingma2014adam} and derives its name from adaptive moment estimation. 
It utilizes both the Adaptive Gradient (AdaGrad) Algorithm and Root Mean Square Propagation (RMSProp). 
Adam only requires first-order gradients and little memory but is computationally efficient and works well with high-dimensional parameter spaces. 
As with AdaGrad and RMSProp, Adam utilizes independent per-parameter learning rates that are separately adapted during training. 
Adam stores a moving average of gradients $E[g]_t=\beta_1 E[g]_t + (1-\beta_1)g_t$ with learning rate $\beta_1>0$. 
Like RMSProp, Adam also stores a moving average of squared gradients $E[g^2]_t$ with learning rate $\beta_2>0$. 
The Adam update rule is given as
\begin{equation}
\theta_{t+1} = \theta_t - \dfrac{\alpha}{\sqrt{\hat{E[g^2]_t}}+\epsilon} \hat{E[g]_t}
\end{equation}
where $\hat{E[g^2]_t}=\dfrac{E[g^2]_t}{1-\beta_2^t}$ and $\hat{E[g]_t}=\dfrac{E[g]_t}{1-\beta_1^t}$. The authors recommend learning rates $\beta_1 = 0.9$, $\beta_2 = 0.999$, as well as $\epsilon = 10^{-8}$. 
Adam has been shown to outperform other optimizers in a wide range of non-convex optimization problems. 






## Back-Propagation. 
Gradient-based optimization requires a method for computing a function's gradient. 
For neural nets, the gradient of the loss function with respect to the weights of the network $\nabla_\theta J(\theta)$ is usually computed using the backpropagation algorithm (backprop) introduced in 1986 \cite{rumelhart1985learning}.
Backpropagation calculates the gradient of the loss function with respect to each weight in the network. 
This is done by iterating backward through the layers of the network and repeatedly applying the chain rule. 
The chain rule of calculus is used when calculating derivatives of functions that are compositions of other functions with known derivatives. 
Let $y, z : \mathbb{R} \rightarrow \mathbb{R}$ be functions defined as $y=g(x)$ and $z=f(g(x))=f(y)$. By the chain rule
\begin{equation}
\dfrac{dz}{dx} = \dfrac{dz}{dy} \dfrac{dy}{dx}
\end{equation}
Generalizing further, let $x\in\mathbb{R}^m, y\in\mathbb{R}^n$, and define mappings $g : \mathbb{R}^m \rightarrow \mathbb{R}^n$ and $f : \mathbb{R}^n \rightarrow \mathbb{R}$. If $y=g(x)$ and $z=f(y)$, then the chain rule is
\begin{equation}
\dfrac{\partial z}{\partial x_i} = \sum_j \dfrac{\partial z}{\partial y_j} \dfrac{\partial y_j}{\partial x_i}
\end{equation}
which can be written in vector notation as
\begin{equation}
\nabla_x z = (\dfrac{\partial y}{\partial x})^\top \nabla_y z
\end{equation}
where $\dfrac{\partial y}{\partial x}$ is the $n\times m$ Jacobian matrix of $g$. 
Backpropagation is often performed on tensors and not vectors.
However, backpropagation with tensors is still performed the same way by multiplying Jacobians by gradients.
Backpropagation with tensors can be performed by flattening a tensor into a vector, performing backprop on the vector, and then reshaping the vector back into a tensor. 
Let $X$ and $Y$ be tensors and $Y=g(X)$ and $z=f(Y)$. The chain rule for tensors is
\begin{equation}
\nabla_x z = \sum_j (\nabla_x Y_j) \dfrac{\partial z}{\partial Y_j}
\end{equation}

By recursively applying the chain rule, a scalar's gradient can be expressed for any node in the network that produced it. 
This is done recursively starting from the output layer and going back through the layers of the network to avoid having to store subexpressions of the gradient or recompute them several times. 






## Activation Function. 
The activation function is what adds nonlinearity to a neural net. Choosing an appropriate activation function depends on the specific problem. To compute the gradient, the activation function must be differentiable. 
Sigmoid functions like the logistic function are usually used, but other functions such as the hyperbolic tangent function $tanh$ can also be used. 
The derivative of the logistic function is close to $0$ except in a small neighborhood around $0$. At each backward step, the $\delta$ is multiplied by the derivative of the activation function. 
The gradient will therefore approach $0$, and thus produce extremely slow learning. 
This is known as the vanishing gradient problem. 
For this reason, the rectified linear unit (ReLU) is the default recommendation for activation function in modern deep neural nets \cite{goodfellow2016deep}. 
ReLU is a ramp function defined as $ReLU(x) = \max \{0, x\}$.
The derivative of the ReLU function is defined as 
\begin{equation} 
ReLU'(x) = 
 {\begin{cases}
 0 & if\ x<0 \\
 1 & if\ x>0 \\
 \end{cases} }
\end{equation}
The derivative is undefined for $x=0$, but it has subdifferential $[0,1]$, and it conventionally takes the value $ReLU'(0)=0$ in practical implementations. 
Since ReLU is a piecewise linear function it optimizes well with gradient-based methods.


ReLU suffers from what is known as the ``dying ReLU problem" where a large gradient could cause the weights of a node to update such that the node will never output anything but $0$. 
Such nodes will not discriminate against any input, and are effectively ``dead". 
This problem can be caused by unfortunate weight initialization or a too-high learning rate. 
To combat the dying ReLU problem generalizations of the ReLU function like the Leaky ReLU (LReLU) activation function has been proposed \cite{goodfellow2016deep}. 
Leaky ReLU allows a small ``leak" for negative values proportional to some slope coefficient $a$, e.g., $a=0.01$, determined before training. 
This allows small gradients to travel through inactive nodes. 
Leaky ReLU will slowly converge even on randomly initialized weights, but can also reduce performance in some applications \cite{goodfellow2016deep}. 








## Regularization. 
Online reinforcement learning is less prone to overfitting than traditional supervised learning approaches, as it does not use limited sets of training data. 
Nevertheless, overfitting is still a potential risk. 
Stochasticity does not necessarily prevent overfitting \cite{zhang2018study}. 
As deep reinforcement learning is applied to critical domains such as healthcare, finance, and energy infrastructure, the model must generalize out-of-sample. 
The test performance of deep RL agents that perform optimally during training has been shown to vary greatly \cite{zhang2018study}. 
Separate training and testing sets with statistically tied data are recommended when training deep RL agents \cite{zhang2018study}. 
Regularization of connection weights lowers the complexity of the network and can mitigate the risk of overfitting \cite{marsland2011machine}. 
Weight decay is frequently used to regularize neural net loss functions by adding the sum of squares of the weights times a constant weight decay parameter $wd \in (0,1]$ \cite{goodfellow2016deep}. As the weight decay parameter $wd$ increases, larger weights are punished more harsher. 
Dropout is a regularization strategy that has been shown to reduce the risk of overfitting by randomly eliminating non-output nodes and their connections from an ensemble of sub-networks during training \cite{srivastava2014dropout}. 




### Batch Normalization. 
Deep neural networks are sensitive to initial random weights and hyperparameters. 
When updating the network, all weights are updated using a loss estimate under the false assumption that weights in the prior layers are fixed. 
In practice, all layers are updated simultaneously. 
Therefore, the optimization step is constantly chasing a moving target. 
The distribution of inputs during training is forever changing. 
This is known as internal covariate shift and it makes the network sensitive to initial weights and slows down training by requiring lower learning rates. 


Batch normalization (batchnorm) is a method of adaptive reparametrization used to train deep networks. 
It was introduced in 2015 by researchers from Google \cite{ioffe2015batch} to help stabilize and speed up training deep neural networks by reducing internal covariate shift. 
Batch normalization normalizes the output distribution to be more uniform across dimensions by standardizing the activations of each input variable for each mini-batch. 
Standardization is rescaling the data to be standard Gaussian, i.e., zero-mean unit variance. 
The following transformation is applied to a mini-batch of activations to standardize it 
\begin{equation}
\hat{x}^{(k)}_{norm} = \dfrac{x^{(k)} - \mathbb{E}[x^{(k)}]}{\sqrt{Var[x^{(k)}] + \epsilon}}
\end{equation}
where the $\epsilon>0$ is a small number such as $10^{-8}$ added to the denominator for numerical stability. 
Normalizing the mean and standard deviation can, however, reduce the expressiveness of the network \cite{goodfellow2016deep}. 
Applying a second transformation step to the mini-batch of normalized activations restores the expressive power of the network
\begin{equation}
\tilde{x}^{(k)} = \gamma \hat{x}^{(k)}_{norm} + \beta 
\end{equation}
where $\beta$ and $\gamma$ are learned parameters that adjust the mean and standard deviation, respectively. This new parameterization is easier to learn with gradient-based methods. 
Batch normalization is usually inserted after fully connected or convolutional layers and before activation functions. 
It speeds up learning and reduces the strong dependence on initial parameters. 
Batch normalization can also have a regularizing effect and in some cases eliminate the need for dropout. 





## Universal Approximation Theorem. 
The universal approximation theorem states that any continuous function between two Euclidean spaces can be approximated by a neural network with a linear output layer and at least one hidden layer with a ``squashing" activation function, provided the network has enough hidden nodes \cite{goodfellow2016deep}
If the activation function in the hidden layer is linear, then the network is equivalent to a network without hidden layers since linear functions of linear functions are themselves linear. 
Despite this ability to represent almost any function with only one hidden layer, it may be easier or even required to approximate more complex functions using deeply-layered ANNs \cite{sutton2018reinforcement}.
The class of ML algorithms that use neural nets with multiple layers is known as deep learning \cite{goodfellow2016deep}. Combining deep learning with reinforcement learning is known as deep reinforcement learning. 
The architecture of neural nets carries an a priori algorithmic preference, known as inductive bias. 
For certain problems, specific network architectures can significantly outperform more deeply layered FNNs. 
E.g., convolutional neural networks perform well when dealing with computer vision tasks, while Long Short Term Memory performs well when dealing with sequence data. 
A neural net inductive bias should be compatible with the bias of the problem it is solving if it is to generalize well out-of-sample. 



If a feedforward network is trained for sequential data, the number of weights can become extremely high because FNNs do not remember past inputs and has no built-in concept of time. 
Therefore, FNNs are not well suited to forecasting time series data. 
Researchers have recently made significant advances in developing neural network architectures that are more suitable for sequential data. 




## Convolutional Neural Networks. 
Convolutional Neural Networks (CNNs) define a class of neural nets that are specialized to process data with a known, grid-like topology such as time-series data (1-dimensional) or images (2-dimensional) \cite{goodfellow2016deep}. Convolutional neural networks have had a profound impact on fields like computer vision \cite{goodfellow2016deep}, and are used in several successful deep RL applications \cite{mnih2013playing}\cite{hausknecht2015deep}\cite{lillicrap2015continuous}. 
2-dimensional CNNs are good at detecting patterns in images, while 1-dimensional CNNs have been successfully used for time series forecasting \cite{goodfellow2016deep}.


A CNN is a neural net that applies convolution instead of general matrix multiplication in at least one layer. 
A convolution is a form of integral transform defined as the integral of the product of two functions after one is reflected about the y-axis and shifted 
\begin{equation}
s(t)=(x*w)(t) = \int x(a)w(t-a)da
\end{equation}
where $x(t)\in\mathbb{R}$ and $w(t)$ is a weighting function. 


The convolutional layer takes the input $x$ with its preserved spatial structure. 
The weights $w$ are given as filters that always extend the full depth of the input volume, but are smaller than the full input size. 
Convolutional neural nets utilize weight sharing by applying the same filter across the whole input. 
The filter slides across the input and convolves the filter with the image. 
It computes the dot product at every spatial location, which makes up the activation map, i.e., the output. 
This can be done using different filters to produce multiple activation maps. 
The way the filter slides across the input can be modified. 
The stride specifies how many pixels the filter moves every step. 
It is common to zero pad the border if the stride is not compatible with the size of the filter and the input. 
After the convolutional layer, a nonlinear activation function is applied to the activation map. 
Convolutional networks may also include pooling layers that reduce the dimension of the data. A larger stride also gives a similar downsampling effect. 









## Recurrent Neural Networks. 
Recurrent Neural Networks (RNNs) define a class of neural nets that allows connection between nodes to create cycles so that outputs from one node affect inputs to another. 
This enables the networks to exhibit temporal dynamic behavior. 
They scale far better than feedforward networks for longer sequences and are well-suited for processing sequential data. 



RNNs generate a sequence of hidden states $h_t$. 
The hidden states enable weight sharing that allows the model to generalize over examples of various lengths. 
Recurrent neural networks are functions of the previous hidden state $h_{t-1}$ and the input $x_t$ at time $t$. 
The hidden units in a recurrent neural network are often defined as a dynamic system $h^{(t)}$ driven by an external signal $x^{(t)}$
\begin{equation}
h^{(t)}=f(h^{(t-1)}, x^{(t)}; \theta)
\end{equation}

Hidden states $h_t$ are utilized by RNNs to summarize problem-relevant aspects of the past sequence of inputs up to $t$ when forecasting future states based on previous states. 
Since the hidden state is a fixed length vector it will be a lossly summary. 
The forward pass is sequential and cannot be parallelized. 
Backprop uses the states computed in the forward pass to calculate the gradient. 
The backprop algorithm used on unrolled RNNs is called backpropagation through time (BPTT). 
All nodes that contribute to an error should be adjusted. For an unrolled RNN, this means that nodes far back in the calculations should also be adjusted. 
Truncated backpropagation through time that only backpropagates for a small number of backward steps can be used to save computational resources, at the cost of introducing bias. 


Every time the gradient backpropagates through a vanilla RNN cell it is multiplied by the transpose of the weights. A sequence of vanilla RNN cells will therefore multiply the gradient with the same factor multiple times. 
If $x>1$ then $\lim_{n\rightarrow \infty}x^n = \infty$, and if $x<1$ then $\lim_{n\rightarrow \infty}x^n = 0$. 
If the largest singular value of the weight matrix is $>1$ the gradient will increase exponentially as it backpropagates through the RNN cells. If the largest singular value is $<1$ the opposite happens where the gradient will shrink exponentially. 
For the gradient of RNNs, this will result in either exploding or vanishing gradients. 
This is why vanilla RNNs trained with gradient-based methods do not perform well, especially when dealing with long-term dependencies. Research by Bengio et al. from 1994 \cite{bengio1994learning} presents both theoretical and experimental evidence to support this conclusion. 
Scaling gradients using gradient clipping can solve the problem of exploding gradients. 
For vanishing gradients, however, the whole architecture of the recurrent network needs to be changed. This is currently a hot topic of research \cite{goodfellow2016deep}. 



### Long Short-Term Memory. 
Long short-term memory (LSMT) is a form of gated RNN that is designed to have better gradient flow properties to solve the problem of vanishing and exploding gradients. 
LSTMs were introduced in 1997 \cite{hochreiter1997long}, and are traditionally used in natural language processing \cite{goodfellow2016deep}. 
Recently, LSMT networks have been successfully applied to financial time series forecasting \cite{siami2018forecasting}. 

Although new architectures like transformers has shown impressive performance in natural language processing and computer vision, LSTMs are still viewed as the state-of-the-art time seires forecasting method. 


LSTMs are composed of a cell and four gates $f, i, g, o$. 
The gates regulate the flow of information to and from the cell. 
While vanilla RNNs only had one hidden state, LSTMs maintain two hidden states at every time step. One is $h_t$ which is similar to the hidden state of vanilla RNNs, and the second is $c_t$ which is the cell state that gets kept inside the network. 
The cell state runs through the LSTM cell with only minor linear interactions. 
The gates regulate what information is passed to the cell state and hidden state. 
When the gradient flows backward it backpropagates from $c_t$ to $c_{t-1}$ and there is only elementwise multiplication by the $f$ gate and no multiplication with the weights. 
Since the LSTMs backpropagate from the last hidden state through the cell states backward it is only exposed to one $tanh$ nonlinear activation function. Otherwise, the gradient is relatively unimpeded. 
Therefore, LSTMs handle long-term dependencies without the problem of exploding or vanishing gradients. 


> Figure showing LSTM network. 



