import pandas as pd
import os
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import itertools

class NaiveModel_mean:
    def __init__(self):
        self.mean = None

    def fit(self, X, y):
        self.mean = sum(y) / len(y)

    def predict(self, X):
        if self.mean is None:
            raise ValueError("Model has not been trained. Call .train() first.")
        return [self.mean] * len(X)


class NaiveModel_median:
    def __init__(self):
        self.median = None

    def fit(self, X, y):
        self.median = np.median(y)

    def predict(self, X):
        if self.median is None:
            raise ValueError("Model has not been trained. Call .train() first.")
        return [self.median] * len(X)


def get_MAPE(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model using MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return mape


def get_RMSE(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model using RMSE
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    # print('y_test', y_test)
    # print('y_pred', y_pred)
    # print(rmse)
    return rmse


def interpretate_cohen_d(cohen_d, model1, model2):
    if cohen_d < 0:
        better_model = model1
    elif cohen_d > 0:
        better_model = model2
    if 0.2 <= np.abs(cohen_d) < 0.5:
        print("> {} is better with Medium effect size".format(better_model))
    elif np.abs(cohen_d) >= 0.5:
        print("> {} is better with Large effect size".format(better_model))
    else:
        print("> {} is better with small effect size".format(better_model))
