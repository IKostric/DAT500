#%%
import os

os.chdir("/home/ivica/DAT500/")
# print(os.getcwd())

os.environ['SPARK_HOME'] = '/home/ivica/spark-3.0.0-preview2-bin-hadoop3.2'

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, random, json, datetime

#%%
from support.fitness import calcDistance
from support.DNA import crossover, mutation

# %%
import findspark
findspark.init()
import pyspark

sc = pyspark.SparkContext(appName ='TSP3')

#%%
def get_initial_population():
    initial_population = []
    for _ in range(population_size):
        dna = np.random.permutation(num_locations).tolist()
        initial_population.append(dna)

    return initial_population


def get_locations():
    with open('data/locations.json', 'r') as f:
        locations = np.array(json.load(f))
    
    return locations


def get_distance(dna):
    distance = calcDistance(locations[dna])
    return dna, distance, 1/distance


def crossoverAndMutation(couple):
    # CROSSOVER
    child = crossover(*couple)

    # MUTATION
    if np.random.rand() < 0.01:
        child = mutation(child)

    return child

def getFitness(rddPop):
    res = np.array(rddPop
        .map(get_distance)
        .collect())

    population, distances, fitness = res.T
    population = np.stack(population)
    fitness = fitness/np.sum(fitness)
    return population, distances, fitness

#%%
locations = get_locations()

#%%
# CONST
num_locations = 100000
num_iterations = 100
population_size = 100
fraction_elites = 0.1

initial_population = get_initial_population()

#%%
start = datetime.datetime.now()

init_population = sc.parallelize(initial_population, 4)
num_elites = int(population_size*fraction_elites)
# init_population.take(10)

population, distances, fitness = getFitness(init_population)
best_ids = np.argsort(distances)
shortest = [distances[best_ids[0]]]


for _ in range(num_iterations):
    
    couple_idx = np.random.choice(population_size, \
        size=(population_size-num_elites, 2), p=fitness.astype(np.float))

    elite = sc.parallelize(np.atleast_2d(population[best_ids[:num_elites]]), 4)
    couples = sc.parallelize(population[couple_idx], 4)
    children = couples\
        .map(crossoverAndMutation)

    new_population = children.union(elite)
    population, distances, fitness = getFitness(new_population)
    best_ids = np.argsort(distances)
    shortest.append(distances[best_ids[0]])

print(datetime.datetime.now() - start)
print("Shortest distance is {}".format(shortest[-1]))

# plt.plot(shortest)
# %%
sc.stop()


# %%
