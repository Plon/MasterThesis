import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
tf.random.set_seed(7)


class LSTM_forecaster():

    def __init__(self, train_set, look_back=1, epochs=100) -> None:
        self.look_back = look_back
        self.num_features = train_set.shape[1] - 1

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler = self.scaler.fit(train_set)

        self.model = Sequential()
        self.model.add(LSTM(4, input_shape=(look_back, (self.num_features+1))))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.fit(train_set, epochs=epochs)

    def fit(self, dataset, epochs=100, batch_size=1, verbose=2) -> None:
        dataset = self.scaler.transform(dataset)
        x, y = create_dataset(dataset, self.look_back)
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict(self, dataset):
        dataset = self.scaler.transform(dataset)
        x, y = create_dataset(dataset, self.look_back)
        y_pred = self.model.predict(x)
        
        temp = np.zeros(shape=(len(y), (self.num_features+1)))
        temp[:,0] = y_pred[:,0]
        y_pred = self.scaler.inverse_transform(temp)[:,0]
        temp[:,0] = y
        y = self.scaler.inverse_transform(temp)[:,0]
        return y_pred, y

    def rmse_score(self, y, predict) -> float:
        return np.sqrt(mean_squared_error(y, predict))


def create_dataset(dataset: np.ndarray, look_back: int): #cuts one element at each edge
    if not isinstance(look_back, int) or look_back < 1 or look_back >= len(dataset):
        raise ValueError("Argument look_back must be integer in range [1, {}]".format(len(dataset)-1))
    
    x, y = [], []
    for i in range(len(dataset)-look_back-1):
        x.append(dataset[i:(i+look_back)])
        y.append(dataset[i+look_back, 0])
    return np.array(x), np.array(y)


def add_trailing_returns(dataset: np.ndarray, freq: list) -> np.ndarray:
    if not (all(isinstance(x, int) for x in freq) or (len(list(x for x in freq if 0 < x < len(dataset))) == len(freq))):
        raise ValueError("Argument freq must be list that only consist of integers within the range [1, {}]".format(len(dataset)))
    if not len(dataset.shape) == 1:
        raise ValueError("Argument dataset expects 1D array.")

    states = dataset.reshape(-1, 1)
    for i in freq:
        newrow = (dataset[i:]/dataset[:-i])-1
        newrow = np.insert(newrow, 0, [0 for _ in range(i)])
        newrow = newrow.reshape(-1, 1)
        states = np.append(states, newrow, axis=1)

    return states


if __name__ == '__main__':
    import yfinance as yf
    instr = yf.Ticker("CL=F") #WTI crude futures
    dataframe = instr.history(period="30d", interval="30m")
    dataset = dataframe['Close'].values
    dataset = dataset.astype('float32')

    #Add trailing 1-tick, 7-tick, 30-tick returns to each state
    states = add_trailing_returns(dataset, [1, 7, 30])
    num_features = states.shape[1]-1

    train_size=2/3
    train, test = train_test_split(states, train_size=train_size, shuffle=False)

    look_back = 1
    model = LSTM_forecaster(train, look_back=1, epochs=10)

    trainPredict, trainY = model.predict(train)
    testPredict, testY = model.predict(test)

    trainScore = model.rmse_score(trainY, trainPredict)
    print('Train score: %.2f RMSE' % (trainScore))    
    testScore = model.rmse_score(testY, testPredict)
    print('Test score: %.2f RMSE' % (testScore))

    plt.plot(testPredict)
    plt.plot(testY)
    plt.show()
