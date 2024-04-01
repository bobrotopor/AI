import pandas as pd
import numpy as np


def mean_from_df(df: pd.DataFrame) -> float:
    """Можно пихать срезы."""
    means = df.mean()
    return sum(means) / len(means)

def norm_by_target(target_mean: float, not_normed_1d_arr: pd.Series):
    not_normed_mean = not_normed_1d_arr.mean()
    norm_coef = target_mean / not_normed_mean
    return not_normed_1d_arr * norm_coef


class LogisticRegressionGD(object):
    
    def init(self):
        self.a = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x @ self.a))      # вектор x нужно дополнить 1 значениями в начале всех факторных признаков
    def predict(self, x):
        return self.sigmoid(x)
    def coefs(self):
        return self.a
    def LogLikelihood(self, x, Y):
        predict = self.predict(x)
        return -sum(Y * np.log2(predict) + (1 - Y) * np.log2(1 - predict)) / self.m
    def CrossEntropy(self, x, Y):
        return (-Y*np.log(self.predict(x)) - (1- Y)*np.log(1 - self.predict(x))).sum()

    def accuracy(self, x, Y):
        return 0
    
    def fit(self, x, Y, alpha = 0.001, epsylon = 0.01, max_steps = 2500, Rtype = "LL"):
        
        x.insert(0, 'ones_col', np.ones(x.shape[0]))
        self.a = np.zeros(x.shape[1]).reshape(x.shape[1], 1)
        self.m = x.shape[0]
        
        x = np.array(x)
        Y = np.array(Y)
        Y = Y.reshape(Y.shape[0], 1)

        steps, errors = [], []
        step = 0
        for _ in range(max_steps):
            if Rtype == "LL":
                new_error = self.LogLikelihood(x, Y)
                dJ = x.T @ (self.predict(x) - Y) / self.m
                self.a -= alpha*dJ
            elif Rtype == "CE":
                new_error = self.CrossEntropy(x, Y)
                #display(new_error)
                dT_a = -x.T @(Y - self.predict(x))
                self.a -= alpha*dT_a
            step += 1
            steps.append(step)
            errors.append(new_error)
            if abs(new_error) < epsylon or len(steps) > max_steps:
               break
        return steps, errors