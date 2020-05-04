#%%
from Base import GA, DNA
import numpy as np
import json

#%%
class SGA(GA):
    def __init__(self, options, locations=None):
        self.options = options
        self.locations = locations
        self.population = None

    def run(self, population=None, island=False):
        num_iterations = self.options.num_iterations
        elite_size, num_parents = self._get_num_elites_and_parents()
        best_fitnesses = []

        if self.locations is None:
            self._get_locations_from_file('data/locations.json')
            
        if population is None:
            population = self._get_initial_population()
        else:
            population = np.array(population)

        normalized_fitness, best_fitness = self._get_fitnesses(population)
        best_fitnesses.append(best_fitness)

        # ga algorithm
        for iteration in range(num_iterations):
            # SELECTION
            new_population = np.empty_like(population)

            # get new population
            couple_idx = self._get_couple_idx(num_parents, normalized_fitness)

            couples = population[couple_idx]
            for i in range(num_parents):
                # CROSSOVER AND MUTATION
                new_population[i*2:(i+1)*2] = DNA.crossoverAndMutation(*couples[i], self.options.mutation_rate)

            # add elites
            best_idx = np.argsort(-normalized_fitness)[:elite_size]
            new_population[num_parents*2:] = population[best_idx]
            

            # DETERMINE FITNESS
            population = new_population
            normalized_fitness, best_fitness = self._get_fitnesses(population)
            best_fitnesses.append(best_fitness)

        
        if island:
            return population[np.argsort(-normalized_fitness)]

        best_idx = np.argmax(normalized_fitness)
        return population[best_idx], best_fitnesses


#%%
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    class options():
        num_iterations = 100
        population_size = 100
        num_locations = 100

        mutation_rate = 0.01
        elite_fraction = 0.1
        locations = '../data/locations.json'
        
    ga = SGA(options)

    ga.run()
    plt.plot(ga.history)
    plt.show()
# %%