import numpy as np


class TradeEnvironment():
    def __init__(self, states, prices, transaction_fraction=0.002, num_prev_observations=10) -> None:
        self.transaction_fraction = float(transaction_fraction)
        num_prev_observations = int(num_prev_observations)
        if num_prev_observations < 1:
            raise ValueError("Argument num_prev_observations must be integer >= 1, and not {}".format(num_prev_observations))
        
        self.states = states 
        self.current_index = 0 
        self.position = 0 # initial empty portfolio
        #self.prices = np.array(list(zip(*states))[0]) # assume prices are in the first collumn
        self.prices = prices # need to have the non-normalized prices to correctly model transaction costs... However, the model perform better with normalized prices as rewards...
        self.observations = np.array([self.newest_observation() for _ in range(num_prev_observations)]) # State vector of previous n observations

    def reset(self) -> np.ndarray:
        """ Reset environment to start and return initial state """
        self.current_index = 0 
        self.position = 0
        self.observations = np.array([self.newest_observation() for _ in range(len(self.observations))]) 
        return self.state()

    def state(self) -> np.ndarray:
        """ Returns the state vector in a flattened format """
        return self.observations.flatten()

    def newest_observation(self) -> np.ndarray:
        """
        Returns:
            state + current position
        """
        return np.insert(self.states[self.current_index], len(self.states[self.current_index]), self.position)

    def reward_function(self, action) -> float: 
        """ R_t = A_{t-1} * (p_t - p_{t-1}) - p_{t-1} * c * |A_{t-1} - A_{t-2}| """ 
        #TODO how to calculate transaction costs with normalized prices
        return (action * (self.states[self.current_index][0] - self.states[self.current_index-1][0])) - (self.states[self.current_index-1][0] * self.transaction_fraction * abs(action - self.position))
        return (action * (self.prices[self.current_index] - self.prices[self.current_index-1])) - (self.prices[self.current_index-1] * self.transaction_fraction * abs(action - self.position))

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        """
        Take step 

        Args:
            action
        Returns:
            new state
            reward
            termination status
            info
        """
        self.current_index += 1 
        assert action <= 1. and action >= -1.
        if self.current_index >= len(self.states): 
            return self.state(), 0, True, {} # The end has been reached
        reward = self.reward_function(action)  
        self.position = action
        self.observations = np.concatenate((self.observations[1:], [self.newest_observation()]), axis=0) # FIFO state vector update
        return self.state(), reward, False, {}
