#Set current working directory to file path
import os 
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import json
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import azureml.core
from azureml.core import Experiment, Workspace, Dataset, Model
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.train.automl import AutoMLConfig
from azureml.automl.core.forecasting_parameters import ForecastingParameters
from azureml.data.datapath import DataPath, DataPathComputeBinding
from azureml.automl.runtime.shared.score import scoring
from azureml.automl.core.shared import constants
from datetime import datetime


### Connect to workspace and create experiment
ws = Workspace.from_config()
experiment_name = 'forecasting'
experiment = Experiment(ws, experiment_name)

### Attach existing AmlCompute
amlcompute_cluster_name = "jonasrha2"
compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
compute_target.wait_for_completion(show_output=True)

### Data
dataset = pd.read_csv('data/Microsoft_Stock.csv')
train, test = train_test_split(dataset, shuffle=False, train_size=0.3)

def_blob_store = ws.get_default_datastore()
data_path = DataPath(datastore=def_blob_store, path_on_datastore='sample_datapath1')
dataset_train = Dataset.Tabular.register_pandas_dataframe(train, data_path, 'train')
dataset_test = Dataset.Tabular.register_pandas_dataframe(test, data_path, 'test')


### Forecasting parameters
target_column_name = "Close"
time_column_name = "Date"
forecast_horizon = 3
freq = 'D'

forecasting_parameters = ForecastingParameters(time_column_name=time_column_name, 
                                               forecast_horizon=forecast_horizon,
                                               freq=freq)


automl_config = AutoMLConfig(task='forecasting',
                             primary_metric='normalized_root_mean_squared_error',
                             blocked_models=["ExtremeRandomTrees", "AutoArima", "Prophet"], #dont know why
                             iterations=100, #speed it up
                             experiment_timeout_minutes=15,
                             enable_early_stopping=True,
                             training_data=dataset_train,
                             label_column_name=target_column_name,
                             compute_target=compute_target,
                             n_cross_validations="auto", # Could be customized as an integer
                             cv_step_size = "auto", # Could be customized as an integer
                             enable_ensembling=False,
                             verbosity=logging.INFO,
                             forecasting_parameters=forecasting_parameters)

#Submit and run experiment, and retrieve best run
remote_run = experiment.submit(automl_config, show_output=True)
remote_run.wait_for_completion()
best_run, fitted_model = remote_run.get_output() 

"""
experiment = Experiment.list(ws, experiment_name=experiment_name)[0]
runs = experiment.get_runs()
newest_run = list(runs)[-1]
_, fitted_model = newest_run.get_output()
"""

### Registering a model
#Enables you to track multiple versions of a model, and retrieve models 
# for inferencing. 
model = Model.register(workspace=ws, 
                        model_name='regression_model',
                        model_path='reg_model.pkl', #local path
                        description='Test regression model')

### Forecasting
#can just use forecast(), however, 
#forecast_quantiles() gives us confidence intervals. 
x_test = test#['Close']
fitted_model.quantiles = [0.01, 0.5, 0.95]
x_pred = fitted_model.forecast_quantiles(test)


### Scoring
horizons = np.ones((len(x_pred))) * forecast_horizon
x_pred[0.5] = x_pred[0.5].replace(np.nan, 0)
#print(np.isnan(x_pred[0.5]).any())
scores = scoring.score_forecasting(x_test['Close'], x_pred[0.5], metrics=[constants.Metric.ForecastMAPE], horizons=horizons)


### Plot results
test_pred = plt.scatter(x_test['Date'], x_pred[0.5], color="b")
test_test = plt.scatter(
    x_test['Date'], x_test['Close'], color="g"
)
plt.legend(
    (test_pred, test_test), ("prediction", "truth"), loc="upper left", fontsize=8
)
plt.show()