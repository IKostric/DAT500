import numpy as np
import json

class GA():
    @staticmethod
    def fitness_func(dna):
        dist = np.roll(dna,-1, axis=0) - dna
        if len(dist.shape) == 1:
            total_distance = np.sum(abs(dist))
        else:
            dist_sqr = np.sum(dist**2, axis=1)
            total_distance = np.sum(np.sqrt(dist_sqr))

        return total_distance

    def crossover(self, dna1, dna2):
        # TODO try to get rid of this two lines
        # in spark
        dna1 = np.array(dna1)
        dna2 = np.array(dna2)

        start, end = self._get_two_points(dna1)
        
        section1, section2 = dna1[start:end], dna2[start:end]
        leftover1 = np.setdiff1d(dna2, section1, assume_unique=True)
        leftover2 = np.setdiff1d(dna1, section2, assume_unique=True)

        child1, child2 = np.empty_like(dna1), np.empty_like(dna2)

        child1[:start] = leftover1[:start]
        child1[start:end] = section1
        child1[end:] = leftover1[start:]

        child2[:start] = leftover2[:start]
        child2[start:end] = section2
        child2[end:] = leftover2[start:]

        return child1, child2

    def mutation(self, dna):
        mtrate = self.options.mutation_rate

        if (mtrate != None) and (np.random.rand() < mtrate):
            start, end = self._get_two_points(dna)
            dna[start:end] = np.flip(dna[start:end])
        return dna

    def crossoverAndMutation(self, dna1, dna2):
        # CROSSOVER
        child1, child2 = self.crossover(dna1, dna2)

        # MUTATION
        child1 = self.mutation(child1)
        child2 = self.mutation(child2)

        return child1, child2

    def _get_two_points(self, dna):
        return np.sort(np.random.choice(len(dna), 2, replace=False))

    def _get_num_elites_and_parents(self):
        pop_size = self.options.population_size
        elite_size = round(pop_size*self.options.elite_fraction*0.5) *2 # make sure it's even
        number_of_parents = (pop_size-elite_size)//2
        return elite_size, number_of_parents

    def _get_locations_from_file(self, filename):
        with open(filename, 'r') as f:
            self.locations = np.array(json.load(f))[:self.options.num_locations]

    def _get_couple_idx(self, num_parents, normalized_fitness):
        return np.random.choice(self.options.population_size, \
                    size=(num_parents, 2), p=normalized_fitness)

    def _get_initial_population(self):
        num_locations = self.options.num_locations
        population_size = self.options.population_size

        initial_population = np.empty((population_size, num_locations), dtype=int)
        for i in range(population_size):
            initial_population[i] = np.random.permutation(num_locations)

        return initial_population

    def _get_fitnesses(self, population):
        fitnesses = np.empty(len(population))
        total = 0
        best = np.inf
        for i, idx in enumerate(population):
            # fitness = evalDistance(pop, self.distance_matrix)
            fitness = self.fitness_func(self.locations[idx])
            fitnesses[i] = 1/fitness
            total += fitnesses[i]

            if fitness < best:
                best = fitness

        return fitnesses/total, best

