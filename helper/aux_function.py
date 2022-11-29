import os, io, time, requests, zipfile
import torch
import numpy as np
from sklearn.datasets import *
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd

def create_data(dataset, data_size, feature_range, output_scale=False, xrange=[-3, 3]):
    e = 9
    """Different kinds of dataset and tasks"""
    # Binary Classification Dataset
    if dataset == "dataset_moon":
        X, Y = make_moons(n_samples=data_size, noise=0.15, random_state=0)

    elif dataset == "dataset_circles":
        X, Y = make_circles(n_samples=data_size, noise=0.01, random_state=0)

    elif dataset == "dataset_synthetic_regression":
        # X_1 = np.random.uniform(-4, -2, size=int(0.4 * data_size))
        # X_2 = np.random.uniform(-2, 2, size=int(0.2 * data_size))
        # X_3 = np.random.uniform(2, 4, size=int(0.4 * data_size))
        # X = np.concatenate((X_1, X_2, X_3))
        X = np.random.uniform(xrange[0], xrange[1], size=data_size)
        Y = np.power(X, 3) + np.random.normal(0, e, size=data_size)
        Y = Y.reshape(-1, 1)
        X = X.reshape(-1, 1)

    elif dataset == "linear":
        X = np.random.uniform(-4, 4, size=data_size)
        Y = 10*X+2 + np.random.normal(0, e, size=data_size)
        Y = Y.reshape(-1, 1)
        X = X.reshape(-1, 1)
    elif dataset == "linearv2":
        X = np.random.uniform(2.5, 7.5, size=data_size)
        Y = 10*X+2 + np.random.normal(0, e, size=data_size)
        Y = Y.reshape(-1, 1)
        X = X.reshape(-1, 1)

    elif dataset == "dataset_synthetic_regression_v2":
        X1 = np.random.uniform(-6, -3, size=int(data_size*(1/2)))
        X2 = np.random.uniform(3, 6, size=int(data_size*(1/2)))
        X = np.concatenate((X1, X2))
        Y = np.power(X, 3) + np.random.normal(0, e, size=data_size)
        Y = Y.reshape(-1, 1)
        X = X.reshape(-1, 1)
    elif dataset == "quadratic":
        X1 = np.random.uniform(-6, -3, size=int(data_size * (1 / 2)))
        X2 = np.random.uniform(3, 6, size=int(data_size * (1 / 2)))
        X = np.concatenate((X1, X2))
        #X = np.random.uniform(-6, 6, size=data_size)
        Y = -5*X**2 + 50 + np.random.normal(0, e, size=data_size)
        Y = Y.reshape(-1, 1)
        X = X.reshape(-1, 1)
    elif dataset == "dataset_boston":
        X, Y = load_boston(return_X_y=True)

    elif dataset == "dataset_concrete":
        df = pd.read_excel(
            'http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls')
        X = df.drop(df.columns[-1], axis=1).to_numpy()
        Y = df[df.columns[-1]].to_numpy()

    elif dataset == "dataset_energy":
        df = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx')
        df = df.drop(df.columns[[-1, -2]], axis=1)
        X = df.drop(df.columns[[-1, -2]], axis=1).to_numpy()
        Y = df[df.columns[-1]].to_numpy()

    elif dataset == "dataset_wine_quality":
        df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
                         sep=';')
        X = df.drop('quality', axis=1).to_numpy()
        Y = df['quality'].to_numpy()

    elif dataset == "dataset_naval_propulsion":
        zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI CBM Dataset.zip'
        r = requests.get(zip_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open('UCI CBM Dataset/data.txt'), sep='  ', header=None)
        X = df.iloc[:, :16].to_numpy()
        Y = df.iloc[:, 16].to_numpy()

    elif dataset == "dataset_yacht":
        df = pd.read_fwf('https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data').dropna()
        X = df.iloc[:, :6].to_numpy()
        Y = df.iloc[:, 6].to_numpy()

    x_scaler = None
    y_scaler = None
    if not dataset in ["dataset_synthetic_regression", "linear", "dataset_synthetic_regression_v2", "linearv2", "quadratic"] :
        x_scaler = preprocessing.MinMaxScaler(feature_range=(feature_range[0], feature_range[1]))
        #x_scaler = preprocessing.StandardScaler()
        X = x_scaler.fit_transform(X)

        if output_scale:
            Y_mean = np.mean(Y)
            Y_std = np.std(Y)
            Y = (Y - Y_mean) / Y_std
            y_scaler = [Y_mean, Y_std]
    return X, Y, x_scaler, y_scaler


def accuracy_binary(y_true, y_pred, variance):
    y_pred = np.round(y_pred)
    acc = accuracy_score(y_true, y_pred)
    ave_log_likelihood = float(variance.sum() / y_pred.shape[0])
    return acc, ave_log_likelihood


def NLL_regression(y_true, y_pred, y_cov):
    y_true = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true.squeeze()
    y_pred = y_pred.numpy() if isinstance(y_pred, torch.Tensor) else y_pred.squeeze()
    y_cov = y_cov.numpy().squeeze() if isinstance(y_cov, torch.Tensor) else y_cov.squeeze()
    s1 = np.sum((y_true - y_pred) ** 2 / y_cov)
    s2_1 = y_true.shape[0] * np.log(2 * np.pi)
    s2_2 = np.sum(np.log(y_cov))
    result = (s1 + s2_1 + s2_2) * 0.5 / y_true.shape[0]
    print(result)
    #result = - np.mean(-0.5 * np.log(2 * np.pi * (y_cov)) - 0.5 * (y_true - y_pred) ** 2 / (y_cov)) # von HLA
    #result = np.mean(0.5*np.log(y_cov) + 0.5*((y_true - y_pred) ** 2 / y_cov)) + 5
    #print(result)
    return result


def rmse_regression(y_true, y_pred, variance):
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
    avg_NLL = NLL_regression(y_true, y_pred, variance)
    return rmse, avg_NLL


def evaluate(y_true, y_pred, variance, type):
    if type == "binary_classification":     # binary classification
        return accuracy_binary(y_true, y_pred, variance)
    elif type.endswith("regression"):               # regression
        return rmse_regression(y_true, y_pred, variance)

def train_and_evaluate(model, X_train, y_train, X_test, y_test, type):
    start = time.time()
    model.train(X_train, y_train)
    end = time.time()
    train_time = end - start
    y_pred, y_cov = model.predict(X_test)
    metric, avg_NLL = evaluate(y_test, y_pred, y_cov, type)
    return train_time, metric, avg_NLL


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def mean_std(rmse, nll, time):
    rmse_mean = np.mean(rmse, axis=0)
    rmse_std = np.std(rmse, axis=0)
    nll_mean = np.mean(nll, axis=0)
    nll_std = np.std(nll, axis=0)
    time_mean = np.mean(time, axis=0)
    time_std = np.mean(time, axis=0)
    return [rmse_mean, rmse_std, nll_mean, nll_std, time_mean, time_std]


def get_toggle_value(t, n):
    if t.value == "Eins":
        return 1
    elif t.value == "Zwei":
        return 2
    elif t.value == "Drei":
        return 3
    else:
        return n

