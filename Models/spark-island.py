#%%
import os

os.chdir("/home/ivica/DAT500/Models")
# print(os.getcwd())

os.environ['SPARK_HOME'] = '/home/ivica/spark-3.0.0-preview2-bin-hadoop3.2'

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, random, json, datetime

#%%
from SGA import SGA

# %%
import findspark
findspark.init()
import pyspark

sc = pyspark.SparkContext(appName ='TSP3')


#%%
sc.addPyFile('SGA.py')

#%%
def get_locations():
    with open('../data/locations.json', 'r') as f:
        locations = np.array(json.load(f))
    
    return locations


#%%
# CONST
class options():
    num_locations = 100
    num_iterations = 100
    population_size = 100
    fraction_elites = 0.1

#%%
sga = SGA(options)
sga.locations = get_locations()

#%%
start = datetime.datetime.now()

workers = sc.parallelize([sga]*4, 4)
workers.foreach(lambda x: x.run())

print(datetime.datetime.now() - start)

# plt.plot(shortest)
# %%
sc.stop()


# %%
