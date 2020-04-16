from mrjob.job import MRJob, MRStep
from mrjob.protocol import TextProtocol
from Base import GA, DNA
import numpy as np
import json


class SparkGlobal(MRJob, GA):
    FILES = ['../data/locations.json', '../Models/Base.py']

    OUTPUT_PROTOCOL = TextProtocol

    def configure_args(self):
        super(SparkGlobal, self).configure_args()
        self.add_passthru_arg('-p', '--population-size', default=10, type=int)
        self.add_passthru_arg('-n', '--num-iterations', default=10, type=int)
        self.add_passthru_arg('-l', '--num-locations', default=10, type=int)

        self.add_passthru_arg('-e', '--elite_fraction', default=0.1, type=float)
        self.add_passthru_arg('-m', '--mutation_rate', default=0.01, type=float)


    def spark(self, input_path, output_path):
        # import findspark
        # findspark.init()
        import pyspark

        sc = pyspark.SparkContext(appName ='TSPGlobal')

        self._get_locations_from_file('locations.json')

        # constants
        num_iterations = self.options.num_iterations
        mrate = self.options.mutation_rate
        elite_size, num_parents = self._get_num_elites_and_parents()

        # Broadcasts
        ga = sc.broadcast(DNA)
        locations = sc.broadcast(self.locations)

        # spark helper functions
        def get_distance(dna):
            distance = ga.value.fitness_func(locations.value[dna])
            return dna, distance, 1/distance

        def breed(couple):
            children = ga.value.crossoverAndMutation(*couple, mrate)
            for child in children:
                yield child

        def unwrap_mat(dist):
            population, distances, fitness = dist.T
            population = np.stack(population)
            fitness = fitness/np.sum(fitness)
            return population, distances, fitness

        best_fitnesses = np.empty(num_iterations+1)

        # initialize populations
        initial_population = self._get_initial_population()
        init_population = sc.parallelize(initial_population)

        dist = np.array(init_population
                    .map(get_distance)
                    .collect())

        population, distances, fitness = unwrap_mat(dist)
        best_ids = np.argsort(distances)
        shortest = [distances[best_ids[0]]]

        for _ in range(num_iterations):
            
            couple_idx = self._get_couple_idx(num_parents, fitness.astype(np.float))

            elite = sc.parallelize(np.atleast_2d(population[best_ids[:elite_size]]))
            couples = sc.parallelize(population[couple_idx])
            children = couples\
                .flatMap(breed)
            new_population = children.union(elite)
            dist = np.array(new_population
                    .map(get_distance)
                    .collect())

            population, distances, fitness = unwrap_mat(dist)
            best_ids = np.argsort(distances)
            shortest.append(distances[best_ids[0]])

        sc.parallelize([(i, population[best_ids[0]].tolist(), short) for i, short in enumerate(shortest)]).saveAsTextFile(output_path)

        sc.stop()

if __name__ == '__main__':
    SparkGlobal.run()