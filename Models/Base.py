import numpy as np
import json, random

class DNA():
    """ Class contains all necessary methods for breeding and fitness evaluation.
    """

    @staticmethod
    def fitness_func(locations):
        """ Static method for fitness evaluation. Calculates the sum of euclidian distances
            between each location.

        Arguments:
            locations {list} -- list of locations in order of travel.

        Returns:
            [float] -- Total distance travelled.
        """
        dist = np.roll(locations,-1, axis=0) - locations
        if len(dist.shape) == 1:
            total_distance = np.sum(abs(dist))
        else:
            dist_sqr = np.sum(dist**2, axis=1)
            total_distance = np.sum(np.sqrt(dist_sqr))

        return total_distance


    @staticmethod
    def crossover(dna1, dna2):
        """ Calculates crossover (OX). Selects substring from one parent and fills in the
            rest from the other, with making sure that every index only apears once.

        Arguments:
            dna1 {list} -- list of indices for parent 1
            dna2 {list} -- list of indices for parent 2

        Returns:
            [tuple] -- tuple containing indeices of two children
        """

        dna1 = np.array(dna1)
        dna2 = np.array(dna2)

        start, end = DNA.get_two_points(len(dna1))
        
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


    @staticmethod
    def mutation(dna, mrate):
        """ Do a mutation on one individual given mrate.

        Arguments:
            dna {list} -- Indices of an individual
            mrate {float} -- mutation rate, chance for a mutation to happen.

        Returns:
            list -- List of indices after mutation.
        """
        if (np.random.rand() < mrate):
            start, end = DNA.get_two_points(dna)
            dna[start:end] = np.flip(dna[start:end])
        return dna


    @staticmethod
    def crossoverAndMutation(dna1, dna2, mrate=None):
        """ Helper method for combinig both crossover and mutation of parents.

        Arguments:
            dna1 {list} -- List of indices of parent 1.
            dna2 {list} -- List of indices of parent 2.

        Keyword Arguments:
            mrate {float} -- mutation rate (default: {None})

        Returns:
            [tuple] -- tuple containing two children after crossover and mutaion 
                        operators.
        """
        # CROSSOVER
        child1, child2 = DNA.crossover(dna1, dna2)

        # MUTATION
        if (mrate != None):
            child1 = DNA.mutation(child1, mrate)
            child2 = DNA.mutation(child2, mrate)

        return child1, child2


    @staticmethod
    def get_two_points(length):
        """ Helper method for getting two random numbers where 0 < a < b < length.

        Arguments:
            length {int} -- upper bound on selection.

        Returns:
            np.array -- [a, b]
        """
        return np.sort(np.random.choice(length, 2, replace=False))


class GA():
    """ Generic methods that every genetic algorithm uses.
    """

    def _get_num_elites_and_parents(self):
        """ Get the split between number of elites and number of parent. Number of parents 
            should be even. Total population is 2*(num parents)+(num elites).

        Returns:
            tuple -- num elites, num parents
        """
        pop_size = self.options.population_size
        elite_size = round(pop_size*self.options.elite_fraction*0.5) *2 # make sure it's even
        number_of_parents = (pop_size-elite_size)//2
        return elite_size, number_of_parents


    def _get_locations_from_file(self, filename):
        """ Get locations. Needed for calculating fitness. Number of locations in 
            the file should be >= num_locations.

        Arguments:
            filename {str} -- filename to read locations from.
        """

        with open(filename, 'r') as f:
            self.locations = np.array(json.load(f))[:self.options.num_locations]


    def _get_couple_idx(self, num_parents, normalized_fitness):
        """ Roulette wheel selection process. Choose parents randomly.

        Arguments:
            num_parents {int} -- Number of couples to select
            normalized_fitness {list} -- list of normalized fitnesses for every individual in 
                                            the population

        Returns:
            np.array -- new couples array of shape (num_parents, 2) 
        """
        return np.random.choice(self.options.population_size, \
                    size=(num_parents, 2), p=normalized_fitness)


    def _get_initial_population(self):
        """ Generate list of individuals. This is done by permuating indices of locations.

        Returns:
            list -- list of all individuals in the populations
        """

        np.random.seed(random.randint(0, 10000))
        
        num_locations = self.options.num_locations
        population_size = self.options.population_size

        initial_population = np.empty((population_size, num_locations), dtype=int)
        for i in range(population_size):
            initial_population[i] = np.random.permutation(num_locations)

        return initial_population


    def _get_fitnesses(self, population):
        """ Helper function to calculate fitnesses for all individuals in the population.
            Score is calculated by inverse of distance.

        Arguments:
            population {list} -- List of individuals

        Returns:
            tuple -- Tuple of (score, best individual)
        """
        fitnesses = np.empty(len(population))
        total = 0
        best = np.inf
        for i, idx in enumerate(population):
            # fitness = evalDistance(pop, self.distance_matrix)
            fitness = DNA.fitness_func(self.locations[idx])
            fitnesses[i] = 1/fitness
            total += fitnesses[i]

            if fitness < best:
                best = fitness

        return fitnesses/total, best

