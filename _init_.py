import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cPickle as pkl
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def dobro(x):
    return x*2

print("dobro do numero e : ", dobro(52))
