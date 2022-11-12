import numpy as np
import torch
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from sklearn.preprocessing import MinMaxScaler


class TradeEnvironment():
    def __init__(self, trades, transaction_fraction=0.002, num_prev_observations=10) -> None:
        self.transaction_fraction = float(transaction_fraction)
        num_prev_observations = int(num_prev_observations)
        if num_prev_observations < 1:
            raise ValueError("Argument num_prev_observations must be integer >= 1, and not {}".format(num_prev_observations))
        
        self.current_index = 0 
        self.timestamps = np.array(trades.index)
        self.year_cycle, self.week_cycle, self.day_cycle = self.time_cycle()

        #Normalize prices
        self.prices = np.array(trades['Close']).reshape(-1, 1)
        #self.prices = MinMaxScaler().fit_transform(self.prices)
        self.scaler = MinMaxScaler().fit(self.prices)
        self.prices = self.scaler.transform(self.prices)
        self.prices = self.prices.flatten()

        self.current_price = self.prices[self.current_index]
        self.position = 0 #start position is an empty portfolio
        self.returns = [0 for _ in range(10)] # returns

        self.observations = np.array([self.newest_observation() for _ in range(num_prev_observations)]) # State vector of previous n observations


    def time_cycle(self) -> tuple[float, float, float]:
        """ Returns the current position in the year, week, day cycle """
        tstep = self.timestamps[self.current_index]
        year_cycle = np.sin(2*np.pi / ((tstep.day_of_year+1)/366)) if tstep.is_leap_year else np.sin(2*np.pi / ((tstep.day_of_year+1)/365))
        week_cycle = np.sin(2*np.pi / ((tstep.day_of_week+1)/7)) #TODO 5 or 7 trading days for commodities?
        day_cycle = np.sin(2*np.pi / ((tstep.hour+1)/24))
        return year_cycle, week_cycle, day_cycle

    def state(self) -> np.ndarray:
        """ Returns the state vector in a flattened format """
        return self.observations.flatten()

    def newest_observation(self) -> np.ndarray:
        """
        Returns:
            price
            position
            return from previous bar
            average return past 3 bars
            average return past 10 bars
            year cycle
            week cycle
            day cycle
        """
        return np.array([
            self.current_price, 
            self.position, 
            self.returns[-1],
            np.average(self.returns[-3:]), 
            np.average(self.returns[-10:]),
            self.year_cycle,
            self.week_cycle,
            self.day_cycle
        ])

    def reward_function(self, action) -> float: 
        """ R_t = A_{t-1} * (p_t - p_{t-1}) - p_{t-1} * c * |A_{t-1} - A_{t-2}| """  
        return action * (self.prices[self.current_index] - self.current_price) - self.current_price * self.transaction_fraction * abs(action - self.position)

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        """
        Args:
            action
        Returns:
            new state
            reward
            termination status
            info
        """
        self.current_index += 1 #take step

        #Check that action is legal
        #action = action - 1. # maybe better to do it here
        assert action <= 1. and action >= -1.

        #Check that there still is another possible step
        if self.current_index >= len(self.prices):
            return self.state(), 0, True, {}

        #TODO check if the balance is still > 0

        #Reward function      
        reward = self.reward_function(action)  

        #Update state
        #TODO prices can be zero. what to do then? same with negative prices can produce neg number in log
        price_before = self.scaler.inverse_transform([[self.current_price]]).flatten()[0]
        price_now = self.scaler.inverse_transform([[self.prices[self.current_index]]]).flatten()[0]
        self.returns.append(np.log(price_now/price_before))

        self.current_price = self.prices[self.current_index]
        self.position = action
        self.year_cycle, self.week_cycle, self.day_cycle = self.time_cycle()

        # FIFO state vector update
        self.observations = np.concatenate((self.observations[1:], [self.newest_observation()]), axis=0)

        return self.state(), reward, False, {}





if __name__ == '__main__':
    import sys
    sys.path.append('/Users/jonashanetho/Desktop/Masteroppgave/MasterThesis/Bars/')
    from bars import Imbalance_Bars, Bars
    import yfinance as yf
    from networks import ActorNetworkContinuous, ActorNetworkDiscrete, ActorNetworkLSTMDiscrete, ActorNetworkLSTMContinuous, ActorNetwork1DConvolutionalDiscrete, ActorNetwork1DConvolutionalContinuous, ActorNetwork1DConvolutionalLSTMDiscrete, ActorNetwork1DConvolutionalLSTMContinuous, CriticNetwork
    from reinforce import reinforce
    from actor_critic_eligibility_traces import actor_critic
    from deep_q_network import deep_q_network, DQN
    from deep_recurrent_q_network import deep_recurrent_q_network, DLSTMQN, DRQN
    from deep_deterministic_policy_gradient import deep_determinstic_policy_gradient, DDPG_Actor, DDPG_Critic
    from advantage_actor_critic import advantage_actor_critic, A2CLSTMActorDiscrete, A2CLSTMActorContinuous
    from reinforce_baseline import reinforce_baseline
    import plotly.express as px
    import matplotlib.pyplot as plt

    symbol = "CL=F" #WTI crude futures
    instr = yf.Ticker(symbol) 
    hist = instr.history(period="30d", interval="30m")
    bar_ids = Imbalance_Bars("volume").get_all_imbalance_ids(hist)
    hist_bar = hist.loc[bar_ids]

    num_features = 8
    num_prev_obs = 2
    total_num_features = num_features * num_prev_obs

    col_scores = []
    num_runs = 500

    ### REINFORCE
    """
    policy = ActorNetworkDiscrete(observation_space=total_num_features).to(device)
    scores = []

    for _ in range(num_runs):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = reinforce(policy, te)
        if _ % 100 == 0:     
            print(_, sum(score), actions)        
        scores.append(sum(score))
    col_scores.append(scores)
    #"""

    ### REINFORCE with baseline
    #"""
    policy = ActorNetworkDiscrete(observation_space=total_num_features).to(device)
    value_function = CriticNetwork(observation_space=total_num_features).to(device)
    scores = []

    for _ in range(num_runs):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = reinforce_baseline(policy, value_function, te, alpha_policy=1e-3, alpha_vf=1e-5)
        if _ % 100 == 0:     
            print(_, sum(score), actions)        
        scores.append(sum(score))
    col_scores.append(scores)
    #"""

    ### REINFORCE CONTINUOUS ACTION SPACE
    """
    policy = ActorNetworkContinuous(observation_space=total_num_features).to(device)
    scores = []

    num_runs = 3000
    for _ in range(num_runs):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = reinforce(policy, te, alpha=1e-5)
        if _ % 100 == 0: 
            print(_, sum(score), actions)        
        scores.append(sum(score))
    col_scores.append(scores)    
    #"""

    ### REINFORCE LSTM
    """
    policy = ActorNetworkLSTMDiscrete(observation_space=total_num_features, n_layers=2).to(device)
    scores = []

    num_runs = 1000

    for _ in range(num_runs):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = reinforce(policy, te)
        if _ % 100 == 0: 
            print(_, sum(score), actions)  
        scores.append(sum(score))
    col_scores.append(scores)
    #"""

    ### REINFORCE LSTM Continuous Action Space
    """
    policy = ActorNetworkLSTMContinuous(observation_space=total_num_features, n_layers=2).to(device)
    scores = []
    num_runs = 2000
    for _ in range(num_runs):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = reinforce(policy, te, alpha=1e-5)
        if _ % 100 == 0: 
            print(_, sum(score), actions)  
        scores.append(sum(score))
    col_scores.append(scores) 
    #"""

    ### REINFORCE Conv
    """
    policy = ActorNetwork1DConvolutionalDiscrete(observation_space=total_num_features).to(device)
    scores = []
    for _ in range(num_runs):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = reinforce(policy, te)
        if _ % 50 == 0: 
            print(_, sum(score), actions)  
        scores.append(sum(score))
    col_scores.append(scores)    
    #"""
    
    ### REINFORCE Conv Continuous Action Space
    """
    policy = ActorNetwork1DConvolutionalContinuous(observation_space=total_num_features).to(device)
    scores = []

    for _ in range(num_runs):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = reinforce(policy, te, alpha=1e-5)
        if _ % 50 == 0: 
            print(_, sum(score), actions)  
        scores.append(sum(score))
    col_scores.append(scores)    
    #"""

    ### REINFORCE Conv LSTM
    """
    policy = ActorNetwork1DConvolutionalLSTMDiscrete(observation_space=total_num_features, num_lstm_layers=2).to(device)
    scores = []

    for _ in range(num_runs):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = reinforce(policy, te)
        print(actions)
        scores.append(sum(score))
    col_scores.append(scores)
    #"""

    ### REINFORCE Conv LSTM Continuous Action Space
    """
    policy = ActorNetwork1DConvolutionalLSTMContinuous(observation_space=total_num_features, num_lstm_layers=2).to(device)
    scores = []
    num_runs = 1000

    for _ in range(num_runs):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = reinforce(policy, te, alpha=1e-5)
        if _ % 50 == 0: 
            print(_, sum(score), actions)          
        scores.append(sum(score))
    col_scores.append(scores)
    #"""

    ### ACTOR-CRITIC with eligibility traces
    """
    actor = ActorNetworkDiscrete(observation_space=total_num_features).to(device)
    critic = CriticNetwork(observation_space=total_num_features).to(device)
    scores = []

    num_runs = 500

    for _ in range(num_runs):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = actor_critic(actor, critic, te, alpha_actor=1e-1, alpha_critic=1e-1)
        #print(_, sum(score), actions)          
        if _ % 100 == 0: 
            print(_, sum(score), actions)     
        scores.append(sum(score))
    col_scores.append(scores)
    #"""   

    ### ACTOR-CRITIC with eligibility traces CONTINUOUS ACTION SPACE
    """
    actor = ActorNetworkContinuous(observation_space=total_num_features).to(device)
    critic = CriticNetwork(observation_space=total_num_features).to(device)
    scores = []

    num_runs = 1000

    for _ in range(num_runs):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = actor_critic(actor, critic, te, alpha_actor=1e-3, alpha_critic=1e-3)
        if _ % 100 == 0: 
            print(_, sum(score), actions) 
        scores.append(sum(score))
    col_scores.append(scores)
    #"""


    #TODO try with different number of layers, also do same with drqn
    ### ACTOR-CRITIC with batched learning
    """
    #actor = ActorNetworkDiscrete(observation_space=total_num_features).to(device)
    critic = CriticNetwork(observation_space=total_num_features).to(device)
    actor = A2CLSTMActorDiscrete(observation_space=total_num_features).to(device)
    scores = []

    num_runs = 2000

    for _ in range(num_runs):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        #score, actions = advantage_actor_critic(actor, critic, te, alpha_actor=1e-3, alpha_critic=1e-5, batch_size=20)
        score, actions = advantage_actor_critic(actor, critic, te, alpha_actor=1e-3, alpha_critic=1e-3, batch_size=20)
        if _ % 100 == 0: 
            print(_, sum(score), actions)
        scores.append(sum(score))
    col_scores.append(scores)
    #"""   

    ### ACTOR-CRITIC CONTINUOUS ACTION SPACE with batched learning
    """
    actor = A2CLSTMActorContinuous(observation_space=total_num_features).to(device)
    #actor = ActorNetworkContinuous(observation_space=total_num_features).to(device)
    critic = CriticNetwork(observation_space=total_num_features).to(device)
    scores = []

    num_runs = 2000

    for _ in range(num_runs):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = advantage_actor_critic(actor, critic, te, alpha_actor=1e-3, alpha_critic=1e-1, batch_size=50, weight_decay=0)
        if _ % 100 == 0: 
            print(_, sum(score), actions) 
        scores.append(sum(score))
    col_scores.append(scores)
    #"""


    #TODO gradient explodes and favours one action, random exploration like this not feasible for live trading
    ### DQN 
    """
    #q_net = CriticNetwork(observation_space=total_num_features, action_space=3).to(device)
    q_net = DQN(observation_space=total_num_features, action_space=3).to(device)

    scores = []
    num_runs = 3000
    for _ in range(num_runs):
        #te = TradeEnvironment(hist_bar, num_prev_observations=1)
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = deep_q_network(q_net, te, batch_size=50, alpha=1e-5)
        #print(_, sum(score), actions)
        if _ % 50 == 0:
            print(_, sum(score), actions)
            print(torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1)) #Can be useful to detect gradient explotion
        scores.append(sum(score))
    col_scores.append(scores)
    #"""    
    
    ### DRQN 
    """
    q_net = DRQN(observation_space=total_num_features, action_space=3).to(device)
    #q_net = DLSTMQN(observation_space=total_num_features, action_space=3, num_lstm_layers=1).to(device)
    
    scores = []
    num_runs = 3000
    prev_actions = 0
    for _ in range(num_runs):
        #te = TradeEnvironment(hist_bar, num_prev_observations=1)
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = deep_recurrent_q_network(q_net, te, batch_size=50, alpha=1e-4)
        #print(_, sum(score), actions)
        if _ % 50 == 0:
            if _ > 0:
                strat_dist = sum((prev_actions - actions)**2)**0.5/len(actions)
                print("Strat dist: ", strat_dist)
            print(_, sum(score), actions)
            print(torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1)) #Can be useful to detect gradient explotion
            prev_actions = actions
        scores.append(sum(score))
    col_scores.append(scores)
    #"""

    ### DDPG
    # Doesnt work, converges to one action for every state
    """
    actor = DDPG_Actor(observation_space=total_num_features).to(device)
    critic = DDPG_Critic(observation_space=total_num_features).to(device)
    
    scores = []
    num_runs = 2000
    for _ in range(num_runs):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = deep_determinstic_policy_gradient(actor, critic, te, batch_size=10, alpha_critic=1e-3, alpha_actor=1e-3)
        if _ % 50 == 0:# or True:
            print(_, sum(score), actions)
            print(torch.nn.utils.clip_grad_norm_(actor.parameters(), 1)) #Can be useful to detect gradient explotion
            print(torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)) #Can be useful to detect gradient explotion
        scores.append(sum(score))
    col_scores.append(scores)
    #"""

    ### PLOTS
    """
    labels = ["R", "R_CA", "R_LSTM", "R_LSTM_CA", "R_Conv", "R_Conv_CA", "AC", "AC_CA"]
    for s, l in zip(col_scores, labels):
        plt.plot(s, label=l)
    plt.xlabel("Run_nr")
    plt.ylabel("{%} return")
    plt.legend()
    plt.show()
    #"""
    
    #"""
    #fig = px.line(x=np.linspace(1, len(scores), num=len(scores)), y=scores)    
    #fig.show()

    from scipy.signal import savgol_filter
    fig = px.line(x=np.linspace(1, len(scores), num=len(scores)), y=savgol_filter(scores, 9, 2))
    fig.show()

    #fig = px.line(x=np.linspace(1, len(scores), num=len(scores)), y=np.cumsum(scores))
    #fig.show()
    #"""
