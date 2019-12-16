import numpy as np
import pandas as pd
import os

# generate related variables
from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot
# seed random number generator
seed(1)
# prepare data
data1 = 20 * randn(1000) + 100
data2 = data1 + (10 * randn(1000) + 50)
# summarize
print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# plot
pyplot.scatter(data1, data2)
pyplot.show()


os.getcwd()

res = pd.read_csv('data/optim_res1.csv')
params = pd.read_csv('data/params1.csv')

params["param_index"] = params["Unnamed: 0"]

dataset = pd.merge(res,params, on="param_index")

del(dataset["Unnamed: 0"])

dataset["loss"]

groupd = dataset.groupby('param_index')

last_ones = groupd.last()

last_ones.describe()

last_ones.sort_values(by='loss', ascending=True)

last_ones["val_acc"]

pyplot.scatter(last_ones["filters_list"], last_ones["val_acc"])
