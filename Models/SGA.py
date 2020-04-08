#%%
from support.fitness import evalDistance
from support.DNA import crossover, mutation
import matplotlib.pyplot as plt
import numpy as np
import json

test = None

# def getFileContents(filename):
#     with open(filename, 'r') as f:
#         contents = [line for line in f]

#     return contents

#%%
class SGA():
    def __init__(self, options):
        self.options = options
        print(options)

    def run(self, population):
        self.best_fitnesses = []

        # print(population)
        self._get_locations()

        fitness = self._get_finesses(population)

        # ga algorithm
        for _ in range(self.options.num_iterations):
            # SELECTION
            pop_size = self.options.population_size
            elite_size = pop_size//10 # 10% elite
            couple_idx = np.random.choice(pop_size, \
                        size=(pop_size-elite_size, 2), p=fitness)
            couples = population[couple_idx]

            best_idx = np.argsort(-fitness)[:elite_size]
            best = population[best_idx]
            
            new_population = best.tolist()
            for couple in couples:
                # CROSSOVER
                child = crossover(*couple)

                # MUTATION
                if np.random.rand() < 0.01:
                    child = mutation(child)
                new_population.append(child)

            # DETERMINE FITNESS
            population = np.array(new_population)
            fitness = self._get_finesses(population)

        best_idx = np.argmax(fitness)
        self.best = population[best_idx]

    def _get_locations(self):
        with open(self.options.locations, 'r') as f:
            self.locations = json.load(f)

        self.points = np.array(self.locations['locations'])
        self.distance_matrix = np.array(self.locations['distances'])

    def _get_finesses(self, population):
        fitnesses = []
        for pop in population:
            fitness = evalDistance(pop, self.distance_matrix)
            fitnesses.append(fitness)

        # plot
        ind = np.argmin(fitnesses)
        # self._plot(population[ind])
        self.best_fitnesses.append(min(fitnesses))
        # print('Shortest:', self.best_fitnesses[-1])
        # end plot
        
        fitnesses = 1/np.array(fitnesses)
        return fitnesses/np.sum(fitnesses)

    def _plot(self, dna):
        dna = np.pad(dna, (0, 1), 'wrap')
        plt.scatter(*self.points)
        plt.plot(*self.points[:,dna])
        plt.show()


#%%
if __name__ == '__main__':
    num_locations = 100

    class options():
        num_iterations = 10000
        population_size = 1000
        locations = 'data/locations.json'
        
    ga = SGA(options)

    pop = np.empty(shape=(options.population_size, num_locations), dtype=int)
    for i in range(options.population_size):
        pop[i] = np.random.permutation(num_locations)

    ga.run(pop)
    plt.plot(ga.best_fitnesses)
    plt.show()
    ga._plot(ga.best)
    plt.show()
# %%