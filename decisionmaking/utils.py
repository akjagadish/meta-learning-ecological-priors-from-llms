import openml
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
import pandas as pd
import torch
import sys
import os
from os import getenv
from dotenv import load_dotenv
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
load_dotenv()
SYS_PATH = getenv('BERMI_DIR')
PARADIGM_PATH = f"{SYS_PATH}/decisionmaking"


def save_real_data_openML(k=2, method='best', num_points=650):
    # 'OpenML-CTR23 - A curated tabular regression benchmarking suite'
    benchmark_suite = openml.study.get_suite(353)
    data = []
    t = 0

    for task_id in benchmark_suite.tasks:  # iterate over all tasks
        if task_id == 361618:  # ignore this task as it is not loadable
            continue
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        features, targets = task.get_X_and_y()  # get the data

        if (features.shape[1] < 99999) and (not np.isnan(features).any()):

            # Normalize the features and targets
            scaler = preprocessing.MinMaxScaler(
                feature_range=(0, 1)).fit(features)
            features = scaler.transform(features)
            scaler_targets = preprocessing.MinMaxScaler(
                feature_range=(0, 1)).fit(targets.reshape(-1, 1))
            targets = scaler_targets.transform(targets.reshape(-1, 1))

            # Select the best feature
            features = SelectKBest(
                f_regression, k=k).fit_transform(features, targets) if method == 'best' else features[:, np.random.choice(features.shape[1], k, replace=False)]

            if features.shape[0] < num_points:
                xs = [features]
                ys = [targets]
            else:
                xs = np.array_split(features, features.shape[0] // num_points)
                ys = np.array_split(targets, targets.shape[0] // num_points)
            for (x, y) in zip(xs, ys):
                for i in range(x.shape[0]):
                    data.append([x[i].tolist(), y[i], i, t])
                t += 1

            df = pd.DataFrame(
                data, columns=['input', 'target', 'trial_id', 'task_id'])
            df.to_csv(f'data/real_data_dim{k}_method{method}_openML.csv')

        else:
            print('not valid data')


def save_real_data_lichtenberg2017(k=2, method='best', num_points=650):

    data = []
    t = 0
    datasets = torch.load(
        '/u/ajagadish/ermi/decisionmaking/data/dataset_torch.pth')
    for (features, targets) in datasets:  # iterate over all tasks

        if (features.shape[1] < 99999) and (features.shape[1] >= k) and (not np.isnan(features).any()):

            # Normalize the features and targets
            scaler = preprocessing.MinMaxScaler(
                feature_range=(0, 1)).fit(features)
            features = scaler.transform(features)
            scaler_targets = preprocessing.MinMaxScaler(
                feature_range=(0, 1)).fit(targets.reshape(-1, 1))
            targets = scaler_targets.transform(targets.reshape(-1, 1))

            # Select the best feature
            features = SelectKBest(
                f_regression, k=k).fit_transform(features, targets) if method == 'best' else features[:, np.random.choice(features.shape[1], k, replace=False)]

            if features.shape[0] < num_points:
                xs = [features]
                ys = [targets]
            else:
                xs = np.array_split(features, features.shape[0] // num_points)
                ys = np.array_split(targets, targets.shape[0] // num_points)
            for (x, y) in zip(xs, ys):
                for i in range(x.shape[0]):
                    data.append([x[i].tolist(), y[i], i, t])
                t += 1

            df = pd.DataFrame(
                data, columns=['input', 'target', 'trial_id', 'task_id'])
            df.to_csv(
                f'data/real_data_dim{k}_method{method}_lichtenberg2017.csv')

        else:
            print('not valid data')


def induce_pseudo_condition_llm_generated_data(condition='ranked'):

    # load data
    env_name = f'{SYS_PATH}/decisionmaking/data/claude_generated_functionlearningtasks_paramsNA_dim4_data20_tasks7284_run0_procid1_pversionunknown'
    data = pd.read_csv(f'{env_name}.csv')  
    data.input = data['input'].apply(lambda x: np.array(eval(x)))

    #fit a linear model for each task then order the features based on the ranking of the coefficients
    for task in data.task_id.unique():
        df_task = data[data['task_id'] == task]
        if len(df_task) > 0:
            y = df_task['target'].to_numpy()
            X = df_task["input"].to_numpy()
            X = np.stack(X)
            # X = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0) + 1e-6)
            # y = (y - y.min(axis=0))/(y.max(axis=0) - y.min(axis=0) + 1e-6)
            X = (X - X.mean(axis=0))/(X.std(axis=0) + 1e-6)
            y = (y - y.mean(axis=0))/(y.std(axis=0) + 1e-6)

            X_linear = PolynomialFeatures(1, include_bias=True).fit_transform(X)

            # linear regression from X_linear to y
            linear_regresion = sm.OLS(y, X_linear).fit()

            # order the features based on the ranking of the coefficients
            if condition == 'ranking':
                order = np.argsort(np.abs(linear_regresion.params[1:]))[::-1]
                # reorder the features in the data frame
                data.loc[data['task_id'] == task, 'input'] = data.loc[data['task_id'] == task, 'input'].apply(lambda x: str(np.array([x[i] for i in order]).tolist()))
            elif condition == 'direction':
                sign = np.sign(linear_regresion.params[1:])
                # change the sign of the features in the data frame
                data.loc[data['task_id'] == task, 'input'] = data.loc[data['task_id'] == task, 'input'].apply(lambda x: str(np.array([x[i]*sign[i] for i in range(len(sign))]).tolist()))
    
    data.to_csv(f'{env_name}_pseudo{condition}.csv', index=False)
