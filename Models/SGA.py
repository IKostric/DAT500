#%%
from support.fitness import calcDistance
from support.DNA import crossover, mutation
import numpy as np
import json

#%%
class SGA():
    def __init__(self, options):
        self.options = options
        self.locations = None
        print(options)

    def run(self):
        self.best_fitnesses = []

        # print(population)
        if self.locations == None:
            self._get_locations()
            
        population = self._get_initial_population()
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
            self.locations = np.array(json.load(f))

        # self.distance_matrix = np.array(self.locations['distances'])

    def _get_initial_population(self):
        num_locations = self.options.num_locations

        initial_population = []
        for _ in range(self.options.population_size):
            dna = np.random.permutation(num_locations).tolist()
            initial_population.append(dna)

        return np.array(initial_population)

    def _get_finesses(self, population):
        distances = []
        for idx in population:
            # fitness = evalDistance(pop, self.distance_matrix)
            distance = calcDistance(self.locations[idx])
            distances.append(distance)

        ind = np.argmin(distances)
        self.best_fitnesses.append(min(distances))
        
        fitnesses = 1/(np.array(distances)**2)
        return fitnesses/np.sum(fitnesses)


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