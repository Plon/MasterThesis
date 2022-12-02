import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import sys
sys.path.append('/Users/jonashanetho/Desktop/Masteroppgave/MasterThesis/Bars/')
from bars import Imbalance_Bars, Bars


def get_trade_data(instruments: list, period="30d", interval="30m"):
    """ Download trade history """
    hist = [yf.Ticker(i).history(period=period, interval=interval) for i in instruments]
    return hist


def get_imbalance_bars(trade_data: list, imbalance_type="volume"):
    """ Get the imbalance bars """
    timestamps = [Imbalance_Bars(imbalance_type).get_all_imbalance_ids(data) for data in trade_data]
    return timestamps


def get_common_timestamps(timestamps: list) -> np.ndarray:
    """ Gets the common timestamps from trade data """
    timestamps = np.array(list(set(timestamps[0]).intersection(*timestamps)))  
    timestamps = np.sort(timestamps)
    return timestamps


def get_prices(trade_data: list, timestamps: np.ndarray) -> np.ndarray:
    """ Returns the closing prices from timestamps """
    prices = np.array([np.array(data['Close'].loc[timestamps]) for data in trade_data])
    return prices


#TODO prices can be zero. what to do then? same with negative prices can produce neg number in log
def get_returns(prices: np.ndarray, lag=1) -> np.ndarray:
    """ Return log returns p_{t+lag} / p_{t}
        add 0 to start as the returns naturally remove lag num elements """
    if lag >= len(prices[0]):
        return np.zeros(prices.shape)
    returns = np.array([np.log(p[lag:]/p[:-lag]) for p in prices])
    returns = np.insert(returns, [0], np.zeros(lag), axis=1)
    return returns


def normalize_prices(prices: np.ndarray) -> np.ndarray:
    """ Normalize every price series """
    normalized_prices = []
    for p in prices:
        price = p.reshape(-1, 1)
        scaler = MinMaxScaler().fit(price)
        price = scaler.transform(price)
        price = price.flatten()
        normalized_prices.append(price)
    return np.array(normalized_prices)


def get_coordinates(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.array((np.sin(positions), np.cos(positions)))


def get_year_time_cycle_coordinates(timestamps) -> tuple[np.ndarray, np.ndarray]:
    year_pos = np.array([2*np.pi * (((stamp.day_of_year+1)/366) if stamp.is_leap_year else ((stamp.day_of_year+1)/365)) for stamp in timestamps])
    return get_coordinates(year_pos)


def get_week_time_cycle_coordinates(timestamps) -> tuple[np.ndarray, np.ndarray]:
    week_pos = np.array([2*np.pi * ((stamp.day_of_week+1)/7) for stamp in timestamps])
    return get_coordinates(week_pos)


def get_day_time_cycle_coordinates(timestamps) -> tuple[np.ndarray, np.ndarray]:
    day_pos = np.array([2*np.pi * ((stamp.hour+1)/24) for stamp in timestamps])
    return get_coordinates(day_pos)


def get_states(instruments: list, period="30d", interval="30m") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Get prices, returns, and time cycle data and create the state array """
    trade_data = get_trade_data(instruments, period, interval)
    timestamps = get_imbalance_bars(trade_data)
    print(np.array(timestamps))
    timestamps = get_common_timestamps(timestamps)
    print(timestamps)
    prices = get_prices(trade_data, timestamps)
    non_normalized_prices = prices
    returns = np.array([get_returns(prices, lag) for lag in [1, 3, 10]])
    returns = returns.reshape(returns.shape[0]*returns.shape[1], returns.shape[2])
    prices = normalize_prices(prices)
    day = get_day_time_cycle_coordinates(timestamps)
    week = get_week_time_cycle_coordinates(timestamps)
    year = get_year_time_cycle_coordinates(timestamps)

    states = prices
    for arr in [returns, year, week, day]:
        states = np.concatenate((states, arr))
    states = states.T
    return states, non_normalized_prices, timestamps
