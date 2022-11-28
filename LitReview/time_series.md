## Time Series. 
A time series is a stochastic process $\{ X_t \}_{t\in T}$ indexed in time order by integers $t\in T$. 
The observed values of a time series is referred to as the realization of the time series process. 





### Measures of Dependence. 
The marginal distribution function is defined as 
\begin{equation}
F_t (x) = P\{ x_t \leq x \}
\end{equation}

The marginal density function, if it exists, is defined as 
\begin{equation}
f_t (x) = \dfrac{\partial F_t(x)}{\partial x}
\end{equation}


The mean, if it exists, is defined as
\begin{equation}
\mu_{xt} = \mathbb{E}(x_t)=\int_{-\infty}^\infty x f_t (x) dx
\end{equation}


Covariance and correlation measures the extent of linear relationship between two variables. Autocovariance and autorcorrelation measures the linear relationship between lagge values of a time series. 
The autocovariance function is defined as the second moment product
\begin{equation}
\gamma_x(s,t)=cov(x_s, x_t)=\mathbb{E}[(x_s - \mu_s)(x_t - \mu_t)]
\end{equation}
for all $s$ and $t$. Autocovariance measures the linear dependence between two points on the same series observed at different times. 
If $s=t$, the autocovariance reduces to the variance because $\gamma_x (t,t) = \mathbb{E}[(x_t - \mu_t)^2] = var(x_t)$. 

The autocorrelation function (ACF) is defined as 
\begin{equation}
\rho (s,t) =  \dfrac{\gamma(s,t)}{\sqrt{\gamma(s,s)\gamma(t,t)}}
\end{equation}
It measures the linear predictability of the series at time $t$ using only the value $x_s$. 
It can be shown, using the Cauchy-Schwarz inequality, that $\rho (s,t) \in [-1, 1]$, making it convenient to work with. 


The cross-covariance function between two series $x_t$ and $y_t$ is defined as
\begin{equation}
\gamma_{xy}(s,t)=cov(x_s, y_t)=\mathbb{E}[(x_s - \mu_{xs})(y_t - \mu_{yt})]
\end{equation}

The cross-correlation function is defined as
\begin{equation}
\rho_{xy} (s,t) =  \dfrac{\gamma_{xy}(s,t)}{\sqrt{\gamma_{xy}(s,s)\gamma_{xy}(t,t)}}
\end{equation}





## Stationarity. 
If neither the mean $\mu_t=\mu$ nor the autocovariance $\gamma_x(t,j)=\gamma_j$ depend on the date $t$, then the process for $X_t$ is said to be covariance-stationary or weakly stationary. 
A process is strictly stationary if, for any values $j_1, j_2, ..., j_n$ the joint distribution of ($Y_t, Y_{t+j_1}, Y_{t+j_2}, ..., Y_{t+j_n}) depends only on the intervals seperating the dates ($j_1, j_2, ..., j_n$) and not on the date istelf ($t$). 
I.e., a strictly stationary time series is one for which the probabilistic properties does not depend on the time at which the series is observed. Thus, a stationary time series exhibits no trends or seasonality. 
All I.I.D stochastic processes are strictly stationary. 
Every strictly stationary process must be weakly stationary, however not every weakly stationary process is strictly stationary. It is possible to imagine that the mean and autocovariance is not functions of time, but perhaps higher moments such as $E(Y^3_t)$ are. 
The requirements for strictly stationary time series are for most practical applications too strong. Therefore a lot of time series analysis focus on weakly stationary series. 
Stationary time series refers to weakly stationary. 



The autocovariance function of a stationary time series is written as 
\begin{equation}
\gamma(h) = cov(x_t, x_{t+h})= \mathbb{E}[(x_{t} - \mu)(x_{t+h} - \mu)]
\end{equation}


The autocorrelation function of a stationary time series is written as 
\begin{equation}
\rho(h) = \dfrac{\gamma(t, t+h)}{\sqrt{\gamma(t, t)\gamma(t+h,t+h)}} = \dfrac{\gamma(h)}{\gamma(0)}
\end{equation}
where it is known that $\rho(h)\in [-1, 1]$ $\forall h$. 


A linear process, $x_t$, is defined to be a linear combination of white noise variates $w_t$, and is given by 
\begin{equation}
x_t= \mu + \sum_{j=-\infty}^\infty \psi_j w_{t-j}
\end{equation}
where $\sum_{j=-\infty}^\infty |\psi_j| < \infty$. 


A process ${x_t}$ is said to be a Gaussian process if the $n$-dimensional vectors $x = (x_{t_1}, x_{t_2}, ..., x_{t_n})'$, for every collection of time points $t_1, t_2,...,t_n$, and every positive integer $n$, have a multivariate normal distribution. 



## Estimation of Correlation. 
The assumption of stationarity becomes critical with analysis performed using sampled data. 

If a time series is stationary the mean can be estimated by the sample mean
\begin{equation}
\bar{x} = \dfrac{1}{n} \sum_{j=1}^n x_j
\end{equation}

The standard error of the estimate is the square root of $var(\bar{x})$, which is given by
\begin{equation}
var(\bar{x}) = \dfrac{1}{n} \sum_{h=-n}^n (1-\dfrac{|h|}{n}) \gamma_x(h)
\end{equation}


The sample autocovariance function is defined as 
\begin{equation}
\widehat{\gamma}(h) = n^{-1} \sum_{t=1}^{n-h} (x_{t+h} - \bar{x})(x_{t} - \bar{x})
\end{equation}
with $\widehat{\gamma}(-h) = \widehat{\gamma}(h)$ for $h=0,1,...,n-1$. 


The sample autocorrelation function is defined as
\begin{equation}
\widehat{\rho}(h) = \dfrac{\widehat{\gamma}(h)}{\widehat{\gamma}(0)}
\end{equation}





## Autoregression. 
An autoregressive model of order $p$, $AR(P)$, is on the form
\begin{equation}
X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + ... + \phi_p X_{t-p} + \varepsilon_t
\end{equation}
where $X_t$ is stationary, and $\phi_j$ are constants. It is assumed that $\varepsilon_t$ is a Gaussian white noise series. 
If the mean is nonzero, $X_t$ can be replaced by 
\begin{equation}
X_t - \mu = \phi_1 (X_{t-1} - \mu) + \phi_2 (X_{t-2} - \mu) + ... + \phi_p (X_{t-p} - \mu) + \varepsilon_t
\end{equation}
Or a term $\alpha = \mu (1-\phi_1 - ... - \phi_p)$ can be inserted into the original expression. 

The autoregressive operator is defined to be 
\begin{equation}
\phi(B) = 1 + \phi_1 B + \phi_2 B^2 + ... + \phi_p B^p
\end{equation}

Using the autoregressive operator, the $AR(p)$ model can be written as
\begin{equation}
\theta(B)X_t = \varepsilon_t
\end{equation}



The first-order autoregressive model $AR(1)$ is given by $X_t = \phi X_{t-1} + \varepsilon_t$. 
Provided $|\phi|<1$ and $X_t$ is stationary it can be represented as a linear process given by
\begin{equation}
X_t = \sum_{j=0}^\infty \phi^j \varepsilon_{t-j}
\end{equation}
This process is stationary with mean zero. The autocovariance function is given as
\begin{equation}
\gamma(h) = cov(x_t, x_{t+h})= \mathbb{E}[(\sum_{k=0}^\infty \phi^k \varepsilon_{t-k})(\sum_{j=0}^\infty \phi^j \varepsilon_{t+h-j})] = \mathbb{E}[(\varepsilon_{t+h} + ... + \phi^h \varepsilon_{t} + \phi^{t+1} \varepsilon_{t-1} + ...)(\varepsilon_{t} + \phi \varepsilon_{t-1} + ...)] = \sigma_\varepsilon^2 \sum_{j=0}^\infty \phi^{h+j}\phi^j = \sigma_\varepsilon^2 \phi^h \sum_{j=0}^\infty \phi^{2j} = \dfrac{\sigma_\varepsilon^2 \phi^h}{1-\phi^2} 
\end{equation}
The autocorrelation function is given as 
\begin{equation}
\rho(h) = \dfrac{\gamma(h)}{\gamma(0)}
\end{equation}



## Moving Average. 
The moving average model of order $q$, $MA(q)$, is on the form
\begin{equation}
X_t = \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + ... + \theta_q \varepsilon_{t-q} 
\end{equation}
where there are $q$ lags in the moving average and $\theta_j$ for $j=1,2,...,q$ are parameters. It is assumed that $\varepsilon_t$ is a Gaussian white noise series. 

The moving average operator is 
\begin{equation}
\theta(B) = 1 + \theta_1 B + \theta_2 B^2 + ... + \theta_q B^q
\end{equation}

Using the moving average operator, the $MA(q)$ model can be written as
\begin{equation}
X_t = \theta (B) \varepsilon_t
\end{equation}

Unlike the autoregressive process, the moving average process is stationary for any values of the parameters $\theta_j$. 


The first-order moving average model $MA(1)$ is given by $X_t = \varepsilon_t + \theta \varepsilon_{t-1}$. The mean is zero. 
The variance is given as 
\begin{equation}
var(X_t) = \gamma(0) = \mathbb{E}(X_t - \mu)^2 = \mathbb{E}(X_t)^2 = (1+\theta^2)\sigma^2_\varepsilon
\end{equation}
he autocovariance function for $h=1$ is given as
\begin{equation}
\gamma(1) = \mathbb{E}(X_t - \mu)(X_{t-1} - \mu) = \mathbb{E}(X_t)(X_{t-1}) = \theta \sigma_\varepsilon^2
\end{equation}
All higher autocovariances are zero. 
The autocorrelation function is given as 
\begin{equation}
\rho(h) = \dfrac{\theta}{(1+\theta^2)}
\end{equation}
for $h=1$, while $\rho(h)=0$ for $h>1$. 



## ARMA. 
An $ARMA$ process is a mix of an autoregressive and a moving average processes. 
A time series $\{X_t \}_{t\in T}$ is an $ARMA(p,q)$ if it is stationary and 
\begin{equation}
X_t = \phi_1 X_{t-1} + ... + \phi_p X_{t-p} + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + ... + \theta_q \varepsilon_{t-q} 
\end{equation}
where $p$ and $q$ are the autoregressive and the moving average orders, respectively. The $ARMA(p,q)$ model can be written using both the $AR$ and $MA$ operator
\begin{equation}
\phi(B)X_t = \theta (B) \varepsilon_t
\end{equation}


This definition of $ARMA$ model has several problems like 
1. Parameter redundant models. 
2. Stationary $AR$ models that depend on the future. 
3. $MA$ models that are not unique. 
To adress these problems, the definition needs some additional restrictions on the model parameters. 
To adress the first problem, henceforth when an $ARMA(p,q)$ model is referenced it is meant in its simplest form. 
To adress the second problem, some definitions must be made. 
The $AR$ and $MA$ polynomials are defined as 
\begin{equation}
\phi(z) = 1 - \phi_1 z - ... - \phi_p z^p
\end{equation}
\begin{equation}
\theta(z) = 1 + \theta_1 z - ... - \theta_p z^p
\end{equation}
respectively, where $z$ is a complex number. 
For $ARMA$ models it will be required that $\phi(z)$ and $\theta(z)$ have no common factors. 
To adress the third problem, the concept of causality will need to be introduced. 
An $ARMA(p,q)$ model is said to be causal if the time series $\{X_t \}_{t\in T}$ can be written as a one-sided linear process 
\begin{equation}
X_t = \sum_{j=0}^\infty \psi_j \varepsilon_{t-j} = \psi(B) \varepsilon_t
\end{equation}
where $\psi(B) = \sum_{j=0}^\infty \psi_j B^j$ and $\sum_{j=0}^\infty |\psi_j|<\infty$; where $\psi_0=1$. 
An $ARMA(p,q)$ model is causal iff $\phi(z) \neq 0$ for $|z| \leq 1$. The coefficients of the linear process (above) can be determined by solving 
\begin{equation}
\psi(z) = \sum_{j=0}^\infty \psi_j z^j = \dfrac{\theta(z)}{\phi(z)}
\end{equation}
where $|z| \leq 1$. I.e., an $ARMA$ process is causal only when the roots of $\phi(z)$ lie outside the unit circle, that is, $\phi(z)=0$ only when $|z|>1$. 


To adress the problem of uniqueness, the following definition allows an infinite autoregressive representation. 
An $ARMA(p,q)$ model is said to be invertible, if the time series $\{X_t \}_{t\in T}$ can be written as 
\begin{equation}
\pi(B) X_t = \sum_{j=0}^\infty \pi_j X_{t-j} = \varepsilon_t
\end{equation}
where $\pi(B) = \sum_{j=0}^\infty \pi_j B^j$, and $\sum_{j=0}^\infty |\pi_j|<\infty$; where $\pi_0=1$. 
An $ARMA(p,q)$ model is invertible iff $\theta(z)\neq 0$ for $|z| \leq 1$. The coefficients $\pi_j$ given (above) can be determined by solving 
\begin{equation}
\pi(z) = \sum_{j=0}^\infty \pi_j z^j = \dfrac{\phi(z)}{\theta(z)}
\end{equation}
where $|z| \leq 1$. I.e., an $ARMA$ process is invertible only when the roots of $\theta(z)$ lie outside the unit circle, that is, $\theta(z)=0$ only when $|z|>1$. 



## Difference Equations. 

The autocorrelation functions of ARIMA processes are difference equations. 
A homogeneous difference equation of order 1 is on the form
\begin{equation}
u_n - \alpha u_{n-1} = 0 
\end{equation}
where $\alpha \neq 0$ and $n=1,2,...$. To solve the equation it can be written as $u_n = \alpha u_{n-1} = \alpha^n u_0$. This be solved given initial condition $u_0 = c$, namely $u_n = \alpha^n c$. In operator notation the original difference equation can be written as $(1-\alpha B) u_n = 0$, and the polynomial associated with it is $\alpha(z)=1-\alpha z$. The root of this polynomial is $z_0 = 1/ \alpha$, that is $\alpha(z_0)=0$. Therefore, the solution with initial condition $u_0 = c$, is 
\begin{equation}
u_n = \alpha^n c = (z_0^{-1})^n c
\end{equation}
That is, the solution to the difference equation depends only on the intial condition and the inverse of the root of the associated polynomial $\alpha(z)$. 


A homogeneous second order difference equation is on the form
\begin{equation}
u_n - \alpha_1 u_{n-1} - \alpha_2 - u_{n-2} = 0 
\end{equation}
The corresponding polynomial is 
\begin{equation}
\alpha(z)=1-\alpha_1 z - \alpha_2 z^2
\end{equation}
which has two roots $z_1, z_2$. 
For $z_1 \neq z_2$, the solution to the difference equation is 
\begin{equation}
u_n = c_1 z_1^{-n} + c_2 z_2^{-n}
\end{equation}
where $c_1, c_2$ depend on the initial conditions. 
Given two initial conditions $u_0, u_1$, we may solve for $c_1, c_2$ 
\begin{equation}
u_0 = c_1 + c_2
\end{equation}
\begin{equation}
u_1 = c_1 z_1^{-1} + c_2 z_2^{-1}
\end{equation}
where $z_1, z_2$ can be solved in terms of $\alpha_1, \alpha_2$ using the quadratic formula.
When $z_1 = z_2 (=z_0)$ a general solution is 
\begin{equation}
u_n = z_0^{-n} (c_1 + c_2 n)
\end{equation}




## Autocorrelation and Partial Autocorrelation. 
Let $x_t = \theta (B) \varepsilon_t$ be an $MA(q)$ process where $\theta(B)=1+ \theta_1 B + ... + \theta_q B^q$. 
Since $x_t$ is a finite linear combination of white noise terms, the process is stationary with mean 
\begin{equation}
\mathbb{E}(x_t)=\sum_{j=0}^q \theta_j \mathbb{E}(\varepsilon_{t-j}) = 0
\end{equation}
where $\theta_0 = 1$. It has the following autocovariance function 
\begin{equation}
\gamma(h)= cov(x_t, x_{t+h}) = \sigma_\varepsilon^2 \sum_{j=0}^{q-h} \theta_j \theta_{j+h}
\end{equation}
for $h\in [0,q]$. For $h>q$ then $\gamma(h)=0$ (remember that $\gamma(h)=\gamma(-h)$). 
Dividing by $\gamma(0)$ yeilds the autocovariance function of an $MA(q)$ process
\begin{equation}
\rho(h)= \dfrac{\sum_{j=0}^{q-h} \theta_j \theta_{j+h}}{1+\theta^2_1 + ... + \theta^2_q}
\end{equation}
for $h\in [1,q]$. For $h>q$ then $\rho(h)=0$. 


For an $ARMA(p,q)$ model $\phi(B) x_t = \theta(B) \varepsilon_t$, where the zeros of $\phi(z)$ are outside the unit circle, write 
\begin{equation}
x_t=\sum_{j=0}^\infty \psi_j \varepsilon_{t-j} 
\end{equation}
It follows that $\mathbb{E}(x_t)=0$. It has the following autocovariance function 
begin{equation}
\gamma(h)= cov(x_t, x_{t+h}) = \sigma_\varepsilon^2 \sum_{j=0}^\infty \psi_j \psi_{j+h}
\end{equation}
It is possible to solve for $\gamma(h)$ by obtaining a homogeneous difference equation directly in terms $\gamma(h)$. First write
begin{equation}
\gamma(h)= cov(x_t, x_{t+h}) =  \sum_{j=1}^p \phi_j \gamma(h-j) + \sigma_\varepsilon^2 \sum_{j=h}^q \theta_j \psi_{j+h}
\end{equation}






As seen for $MA(q)$ models, the autocorrelation function will be zero for lags greater than $q$. 
Partial autocorrelation is the correlation between variables $x_s, x_t$ with the linear effect of everything in the middle removed. 
Let $\hat{x}_{t+h}$ for $h \geq 2$ denote the regression of $x_{t+h}$ on {$x_{t+h-1}, x_{t+h-2},..., x_{t+1}$}, which is written as
\begin{equation}
\hat{x}_{t+h} = \beta_1 x_{t+h-1} + \beta_2 x_{t+h-2} + ... + \beta_{h-1} x_{t+1} 
\end{equation}
Additionally, let $\hat{x}_{t}$ denote the regression of $x_{t}$ on {$x_{t+1}, x_{t+2},..., x_{t+h-1}$}, which is written as
\begin{equation}
\hat{x}_{t} = \beta_1 x_{t+1} + \beta_2 x_{t+2} + ... + \beta_{h-1} x_{t+h-1} 
\end{equation}
Because of stationarity, the coefficients $\beta_j$ are the same in both expressions. 
The partial autocorrelation function of a stationary process, $x_t$, denoted $\phi_{hh}$ for $h=1,2,...$, is 
\begin{equation}
\phi_{11}=corr(x_t, x_{t+1}) = \rho(1)
\end{equation}
\begin{equation}
\phi_{hh}=corr(x_t - \hat{x}_t, x_{t+h} - \hat{x}_{t+h})
\end{equation}
for $h \geq 2$. Both ($x_t - \hat{x}_t$) and ($x_{t+h} - \hat{x}_{t+h}$) are uncorrelated. 



## Forecasting.
A goal in forecasting can be to predict the value of a variable $X_{t+1}$ based on a set of variables $Y_t$ observed at time $t$. E.g., forecasting $X_{t+1}$ based on its $m$ most recent values, i.e., $Y_t$ would consist of a constant plus $X_t, X_{t-1}, ..., X_{t-m+1}$. 
Let $X^*_{t+1 | t}$ denote a forecast of $X_{t+1}$ based on $Y_t$. 
The loss function is defined to measure the usefulness of the forecast. 
A common choice with convenient results is the quadratic loss function mean squared error 
\begin{equation}
MSE(X^*_{t+1 | t}) = \mathbb{E}(X_{t+1} - X^*_{t+1 | t})^2
\end{equation}
The forecast $g(Y_t)$ that minimizes the MSE is the expectation of $X_{t+1}$ conditional on $Y_t$ 
\begin{equation}
X^*_{t+1 | t} = \mathbb{E}(X_{t+1} | Y_t)
\end{equation}
The MSE of this optimal forecast is 
\begin{equation}
MSE(g(Y_t)) = \mathbb{E}(X_{t+1} - g(Y_t))^2 = \mathbb{E}(X_{t+1} - \mathbb{E}(X_{t+1} | Y_t))^2
\end{equation}


### Forecasts Based on Linear Projection. 
Forecasting $X^*_{t+1 | t}$ based on a linear projection of $Y_t$ has the form 
\begin{equation}
X^*_{t+1 | t} = \alpha' Y_t
\end{equation}
If there is a value for $\alpha$ such that the forecast error $(X_{t+1} - \alpha' Y_t)$ is uncorrelated with $Y_t$
\begin{equation}
\mathbb{E}((X_{t+1} - \alpha' Y_t)Y_t') = 0'
\end{equation}
If this holds, then the forecast $\alpha' Y_t$ is called the linear projection of $X_{t+1}$ on $Y_t$. 
The linear projection produces the smallest mean squared error among the class of linear forecasting rules. 
The following notation will be used to indicate the linear projection of $X_{t+1}$ on $Y_t$ 
\begin{equation}
\hat{P}(X_{t+1} | Y_t) = \hat{X}_{t+1 | t}  = \alpha' Y_t
\end{equation}
Since the conditional expectation offers the best possible forecast, the following holds
\begin{equation}
MSE[\hat{P}(X_{t+1} | Y_t)] \geq MSE[\mathbb{E}(X_{t+1} | Y_t)]
\end{equation}
For most applications a constant term is included in the projection. The symbol $\hat{\mathbb{E}}$ is used to indicate a linear projection on a vector of random variables $Y_t$ with a consant term
\begin{equation}
\hat{\mathbb{E}}(X_{t+1} | Y_t) = \hat{P}(X_{t+1} | 1, Y_t)
\end{equation}



### Properties of Linear Projection. 
The coefficient $\alpha$ can be calculated in terms of the moments of $X_{t+1}$ and $Y_t$
\begin{equation}
\alpha' = \mathbb{E}(X_{t+1}Y_t')[\mathbb{E}(Y_t Y_t')]^{-1}
\end{equation}
The MSE associated with a linear projection is then given by 
\begin{equation}
MSE(\hat{X}_{t+1 | t}) = \mathbb{E}(X_{t+1} - \alpha' Y_t)^2 = \mathbb{E}(X_{t+1})^2 - 2\mathbb{E}(\alpha' Y_t X_{t+1}) + \mathbb{E}(\alpha' Y_t Y_t' \alpha) = \mathbb{E}(X_{t+1})^2 - \mathbb{E}(X_{t+1}Y_t)[\mathbb{E}(Y_t Y_t')]^{-1} \mathbb{E}(Y_t X_{t+1})
\end{equation}
If $Y_t$ includes a constant term, the projection of ($aX_{t+1}+b$) on $Y_t$ (where $a,b$ are deterministic constants) is equal to
\begin{equation}
\hat{P}[(aX_{t+1}+b)|Y_t]=a\cdt \hat{P}(X_{t+1} | Y_t)+b
\end{equation}
which is uncorrelated with $Y_t$ as is required of a linear projection. 



### Linear Projection and Ordinary Least Squares Regression. 
Linear projection is closely related to ordinary least squares regression. A linear regression model relates an observation on $x_{t+1}$ to $y_t$ 
\begin{equation}
x_{t+1} = \beta' y_t + u_t
\end{equation}
Given a sample of $T$ observations on $x$ and $y$, the sample sum of squared residuals is defined as
\begin{equation}
\sum_{t=1}^T (x_{t+1}-\beta' y_t)^2
\end{equation}
The value of $\beta$ that minimizes the sum of squared residuals, denoted $b$, is the ordinary least squares (OLS) estimate of $\beta$. 
The formula for $b$ is 
\begin{equation}
b = [(1/T) \sum_{t=1}^T (y_{t}y_t')^2]^{-1} [(1/T) \sum_{t=1}^T (y_{t}x_{t+1})^2]
\end{equation}
OLS $b$ is constructed using sample moments, while linear projection $\alpha$ is constructed using population moments. 
If the stochastic process $\{ Y_t, X_{t+1} \}$ is weakly stationary and ergodic for second moments, then the sample moments will converge to the population moments as the sample size $T$ goes to infinity. 








## ARIMA. 
Differencing a process $x_t=x_{t-1} + \varepsilon_t$ composed of a nonstationary trend component and a zero-mean stationary component leads to a stationary process $\nabla x_t = \varepsilon_t$. 

The integrated ARMA, or ARIMA, model is a broadening of the class of ARMA models to include differencing. 
A process $x_t$ is said to be $ARIMA(p, d, q)$ if 
\begin{equation}
\nabla^d x_t=(1-B)^d x_t
\end{equation}
is $ARMA(p,q)$. In general the model is written as
\begin{equation}
\phi(B)(1-B)^d x_t = \theta(B)\varepsilon_t
\end{equation}
If $\mathbb{E}(\nabla^d x_t)=\mu$ then the model is written as
\begin{equation}
\phi(B)(1-B)^d x_t = \delta + \theta(B)\varepsilon_t
\end{equation}
where $\delta=\mu(1-\phi_1 - ... - \phi_p)$.

