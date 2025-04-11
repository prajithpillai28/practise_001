import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class ML_algorithms:

    def __init__(self,data):

        self.data=data
        ML_algorithms.split(self, data)

    def split(self,data):
        X_train,X_val=train_test_split(data)

    def logistic(self,Xtrain):
        y_train=LogisticRegression(Xtrain)
        return y_train

