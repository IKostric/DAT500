from mrjob.job import MRJob, MRStep
from Base import GA, DNA
import numpy as np
import json


class MRJobIsland(MRJob, GA):
    FILES = ['../data/locations.json', '../Models/Base.py']

    def configure_args(self):
        super(MRJobIsland, self).configure_args()
        self.add_passthru_arg('-p', '--population-size', default=10, type=int)
        self.add_passthru_arg('-n', '--num-iterations', default=10, type=int)
        self.add_passthru_arg('-l', '--num-locations', default=10, type=int)

        self.add_passthru_arg('-e', '--elite-fraction', default=0.1, type=float)
        self.add_passthru_arg('-m', '--mutation-rate', default=0.01, type=float)

        self.add_passthru_arg('--num-islands', default=3, type=int)
        self.add_passthru_arg('--num-migrations', default=2, type=int)
        self.add_passthru_arg('--migrant-fraction', default=0.3, type=float)


    def mapper_init(self):
        self._get_locations_from_file('locations.json')


    def mapper(self, key, island_info):
        if self.options.step_num == 0:
            population = self._get_initial_population()
            key = "island-{}".format(island_info)
        else:
            population = np.array(island_info[0])
            best_fitnesses = island_info[1]

        normalized_fitness, best_fitness = self._get_fitnesses(population)
        if self.options.step_num == 0:
            best_fitnesses = [best_fitness]

        num_iterations = self.options.num_iterations
        pop_size = self.options.population_size
        elite_size, num_parents = self._get_num_elites_and_parents()

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
            best_idx = np.argsort(-normalized_fitness)
            new_population[num_parents*2:] = population[best_idx[:elite_size]]

            # DETERMINE FITNESS
            population = new_population
            normalized_fitness, best_fitness = self._get_fitnesses(population)
            best_fitnesses.append(best_fitness)


        best_idx = np.argsort(-normalized_fitness)

        if self.options.step_num < self.options.num_migrations:
            # (num islands-1) *  number of migrants per island (2)
            num_other_islands = self.options.num_islands-1
            number_of_migrants = int(((pop_size*self.options.migrant_fraction)
                                    /num_other_islands)) # per island

            migrant_ids = best_idx[:number_of_migrants*num_other_islands] 
            all_migrants = np.random.permutation(population[migrant_ids]).tolist()

            mask = np.ones(pop_size, dtype=bool)
            mask[migrant_ids] = False

            rest_pop = population[mask].tolist()

            for i in range(self.options.num_islands):
                other = "island-{}".format(i)
                if other != key:
                    yield other, (all_migrants[:number_of_migrants], [])
                    all_migrants = all_migrants[number_of_migrants:]
                
            yield key, (rest_pop, best_fitnesses)

        else:
            best = population[best_idx[0]].tolist()
            yield "result", (best, best_fitnesses)


    def reducer(self, key, values):
        if self.options.step_num < self.options.num_migrations:
            population = []
            distances = []
            for pop, dist in values:
                population += pop
                distances += dist
            yield key, (population, distances)
        else:
            best, distances = list(zip(*map(list, values)))
            shortest_distance = []
            for dist in zip(*distances):
                shortest_distance.append(min(dist))

            yield best[np.argmin(dist)], shortest_distance

    def steps(self):
        return [MRStep(mapper_init=self.mapper_init,
                        mapper=self.mapper,
                        reducer=self.reducer)
                    ]*(self.options.num_migrations+1)


if __name__ == '__main__':
    MRJobIsland.run()