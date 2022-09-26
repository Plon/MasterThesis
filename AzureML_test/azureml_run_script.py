from azureml.core import Run
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os, joblib, argparse


#start the experiment
run = Run.get_context()

#Set regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--train-size', type=float, dest='train_size', default=2/3)
args = parser.parse_args()
train_size = args.train_size

#experiment code hoes here
house = pd.read_csv('data/House_Rent_Dataset.csv')
house = house.loc[house['City'] == 'Mumbai']
X, y = house[['Size', 'BHK', 'Bathroom']].values, house['Rent'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

#Train a logistic regression model
model = LinearRegression().fit(X_train, y_train)

#calculate accuracy
y_hat = model.predict(X_test)
mse = mean_squared_error(y_hat, y_test)
run.log('MSE: ', np.float64(mse))

#save a sample of the data
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

#end the experiment
run.complete()
