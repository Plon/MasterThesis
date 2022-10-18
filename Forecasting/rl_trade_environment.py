import numpy as np
import torch
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from sklearn.preprocessing import MinMaxScaler


class TradeEnvironment():
    def __init__(self, trades, transaction_fraction=0.01, num_prev_observations=10) -> None:
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
        #action = action - 1.
        assert action <= 1. and action >= -1.

        #Check that there still is another possible step
        if self.current_index >= len(self.prices):
            return self.state(), 0, True, {}

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
    from reinforce import PolicyNetworkDiscrete, PolicyNetworkContinuous, reinforce
    from actor_critic_eligibility_traces import ActorNetworkContinuous, ActorNetworkDiscrete, CriticNetwork, actor_critic
    import plotly.express as px

    symbol = "CL=F" #WTI crude futures
    instr = yf.Ticker(symbol) 
    hist = instr.history(period="30d", interval="30m")
    bar_ids = Imbalance_Bars("volume").get_all_imbalance_ids(hist)
    hist_bar = hist.loc[bar_ids]

    num_features = 8
    num_prev_obs = 5
    total_num_features = num_features * num_prev_obs

    ### REINFORCE
    """
    policy = PolicyNetworkDiscrete(observation_space=total_num_features).to(device)
    scores = []

    for _ in range(100):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = reinforce(policy, te)
        print(actions)
        scores.append(sum(score))
    #"""

    ### REINFORCE CONTINUOUS ACTION SPACE
    """
    policy = PolicyNetworkContinuous(observation_space=total_num_features).to(device)
    scores = []

    for _ in range(100):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = reinforce(policy, te)
        print(actions)
        scores.append(sum(score))
    
    #"""

    ### ACTOR-CRITIC
    """
    actor = ActorNetworkDiscrete(observation_space=total_num_features).to(device)
    critic = CriticNetwork(observation_space=total_num_features).to(device)
    scores = []

    for _ in range(100):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = actor_critic(actor, critic, te)
        print(actions)
        scores.append(sum(score))
    #"""   

    ### ACTOR-CRITIC CONTINUOUS ACTION SPACE
    """
    actor = ActorNetworkContinuous(observation_space=total_num_features).to(device)
    critic = CriticNetwork(observation_space=total_num_features).to(device)
    scores = []

    for _ in range(100):
        te = TradeEnvironment(hist_bar, num_prev_observations=num_prev_obs)
        score, actions = actor_critic(actor, critic, te)
        print(actions)
        scores.append(sum(score))
    #"""

    ### PLOTS
    """
    fig = px.bar(x=np.linspace(1, len(scores), num=len(scores)), y=scores)    
    fig.show()

    cumulative_scores = np.cumsum(scores)
    fig = px.bar(x=np.linspace(1, len(cumulative_scores), num=len(cumulative_scores)), y=cumulative_scores)
    fig.show()
    #"""
