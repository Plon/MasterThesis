import itertools
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import sys
sys.path.append('/Users/jonashanetho/Desktop/Masteroppgave/MasterThesis/Bars/')
from bars import Imbalance_Bars, Bars


def get_trade_data_yf(instruments: list, period="30d", interval="30m"):
    """ Download trade history """
    hist = [yf.Ticker(i).history(period=period, interval=interval) for i in instruments]
    return hist


def get_imbalance_bars(trade_data: list, imbalance_type="volume"):
    """ Get the imbalance bars """
    timestamps = [Imbalance_Bars(imbalance_type).get_all_imbalance_ids(data) for data in trade_data]
    return timestamps


def get_common_timestamps(timestamps: list) -> np.ndarray:
    """ Gets the common timestamps from trade data """ 
    #TODO this removes duplicates 
    timestamps = np.array(list(set(timestamps[0]).intersection(*timestamps)))  
    timestamps = np.sort(timestamps)
    return timestamps


def get_prices_yf(trade_data: list, timestamps: np.ndarray) -> np.ndarray:
    """ Returns the closing prices from timestamps """
    prices = np.array([np.array(data['Close'].loc[timestamps]) for data in trade_data])
    return prices


#TODO prices can be zero. what to do then? same with negative prices can produce neg number in log
def get_log_returns(prices: np.ndarray, lag=1) -> np.ndarray:
    """ Return log returns p_{t+lag} / p_{t}
        add 0 to start as the returns naturally remove lag num elements """
    if lag >= len(prices):
        return np.zeros(prices.shape)
    returns = np.array(np.log(prices[lag:]/prices[:-lag]))
    returns = np.insert(returns, [0], np.zeros(lag), axis=0)
    return returns


def normalize_prices(prices: np.ndarray) -> np.ndarray:
    """ Return the normalized price series """
    prices = prices.reshape((len(prices), 1))
    scaler = MinMaxScaler().fit(prices)
    prices = scaler.transform(prices)
    prices = prices.flatten()
    return prices


def get_coordinates(positions: np.ndarray) -> np.ndarray:
    return np.array((np.sin(positions), np.cos(positions)))


def get_year_time_cycle_coordinates(timestamps) -> np.ndarray:
    year_pos = np.array([2*np.pi * (((stamp.dayofyear+1)/366) if stamp.is_leap_year else ((stamp.dayofyear+1)/365)) for stamp in timestamps])
    return get_coordinates(year_pos)


def get_week_time_cycle_coordinates(timestamps) -> np.ndarray:
    week_pos = np.array([2*np.pi * ((stamp.dayofweek+1)/7) for stamp in timestamps])
    return get_coordinates(week_pos)


def get_day_time_cycle_coordinates(timestamps) -> np.ndarray:
    day_pos = np.array([2*np.pi * ((stamp.hour+1)/24) for stamp in timestamps])
    return get_coordinates(day_pos)


def get_states_from_yf(instruments: list, period="30d", interval="30m", imb_bars=True, riskfree_asset=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Get prices, returns, and time cycle data and create the state array """
    trade_data = get_trade_data_yf(instruments, period, interval)
    if imb_bars:
        timestamps = get_imbalance_bars(trade_data)
    else:
        timestamps = [np.array(td.index) for td in trade_data]
    timestamps = get_common_timestamps(timestamps)
    
    prices = get_prices_yf(trade_data, timestamps)
    non_normalized_prices = prices
    lag_intervals = [1, 3, 10]
    returns = np.array([get_log_returns(p, l) for p, l in itertools.product(prices, lag_intervals)])
    prices = np.array([normalize_prices(p) for p in prices])
    
    day = get_day_time_cycle_coordinates(timestamps)
    week = get_week_time_cycle_coordinates(timestamps)
    year = get_year_time_cycle_coordinates(timestamps)

    if riskfree_asset:
        riskfree_price = np.zeros((1, len(timestamps),)) # no transaction cost if price is zero
        prices = np.concatenate((riskfree_price, prices))
        riskfree_returns = np.zeros((len(lag_intervals), len(timestamps)))
        returns = np.concatenate((riskfree_returns, returns))

    states = prices
    for arr in [returns, year, week, day]:
        states = np.concatenate((states, arr))
    states = states.T
    return states, non_normalized_prices, timestamps
