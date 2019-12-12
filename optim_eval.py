import pandas as pd
import os

os.getcwd()

params = pd.read_csv('optimizations/params.csv')
results = pd.read_csv('optimizations/optim_results.csv')

params = params[0:len(results)]

merge = pd.merge(params,results)
