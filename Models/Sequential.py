#%%
from Base import GA, DNA
import numpy as np
import json

#%%
class SGA(GA):
    """ Basic sequential genetic algorithm. This is used as a benchmark
    to compare other algorithms to in terms of speed and quality.

    Arguments:
        GA {class} -- inherit from GA class
    """

    def __init__(self, options, locations=None):
        self.options = options
        self.locations = locations
        self.population = None


    def run(self, population=None, island=False):
        """ Run sequential algorithm. At every step do selection, crossover and mutation
        in order to get new generation of individuals.

        Keyword Arguments:
            population {np.array} -- 2D array, every row is an array of indices of locations (default: {None})
            island {bool} -- If island, return entire population so migration can occur,
                             otherwise return result (default: {False})

        Returns:
            tuple -- best individual, history of best scores at every iteration
        """

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
    """ Should be called from Driver.py. This is only for testing purposes.
    """
    class options():
        num_iterations = 100
        population_size = 100
        num_locations = 100

        mutation_rate = 0.01
        elite_fraction = 0.1
        locations = '../data/locations.json'
        
    ga = SGA()
    ga.run()
# %%