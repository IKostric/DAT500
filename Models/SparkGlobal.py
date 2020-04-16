from mrjob.job import MRJob, MRStep
from Base import GA
import numpy as np
import json


class SparkGlobal(MRJob, GA):
    FILES = ['../data/locations.json', '../Models/Base.py']

    def configure_args(self):
        super(SparkGlobal, self).configure_args()
        self.add_passthru_arg('-p', '--population-size', default=10, type=int)
        self.add_passthru_arg('-n', '--num-iterations', default=10, type=int)
        self.add_passthru_arg('-l', '--num-locations', default=10, type=int)

        self.add_passthru_arg('-e', '--elite_fraction', default=0.1, type=float)
        self.add_passthru_arg('-m', '--mutation_rate', default=0.01, type=float)


    def spark(self, input_path, output_path):
        import pyspark
        sc = pyspark.SparkContext(appName ='TSP3')

        self._get_locations_from_file('locations.json')

        # Avoid serialization of the entire object
        locations = self.locations
        fitness_func = self.fitness_func

        # spark helper functions
        def get_distance(dna):
            distance = fitness_func(locations[dna])
            return dna, distance, 1/distance

        def getFitness(dist):
            population, distances, fitness = dist.T
            population = np.stack(population)
            fitness = fitness/np.sum(fitness)
            return population, distances, fitness

        # constants
        num_iterations = self.options.num_iterations
        elite_size, num_parents = self._get_num_elites_and_parents()
        best_fitnesses = np.empty(num_iterations+1)

        # initialize populations
        initial_population = self._get_initial_population()
        init_population = sc.parallelize(initial_population, 4)

        dist = np.array(init_population
                    .map(lambda dna: fitness_func(locations[dna]))
                    .collect())

        # population, distances, fitness = getFitness(dist)
        # best_ids = np.argsort(distances)
        # shortest = [distances[best_ids[0]]]

        # for _ in range(num_iterations):
            
        #     couple_idx = np.random.choice(population_size, \
        #         size=(population_size-num_elites, 2), p=fitness.astype(np.float))

        #     elite = sc.parallelize(np.atleast_2d(population[best_ids[:num_elites]]), 4)
        #     couples = sc.parallelize(population[couple_idx], 4)
        #     children = couples\
        #         .map(crossoverAndMutation)

        #     new_population = children.union(elite)
        #     population, distances, fitness = getFitness(new_population)
        #     best_ids = np.argsort(distances)
        #     shortest.append(distances[best_ids[0]])

        # children.saveAsHadoopFile(output_path,
        #                     'nicknack.MultipleValueOutputFormat')

        sc.stop()