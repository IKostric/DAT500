from mrjob.job import MRJob, MRStep
import numpy as np
import json


class IPGA(MRJob):
    DIRS = ['../data', '../support']

    def configure_args(self):
        super(IPGA, self).configure_args()
        self.add_passthru_arg('-d', '--locations', default='data/locations.json')
        self.add_passthru_arg('-p', '--population-size', default=10, type=int)
        self.add_passthru_arg('-n', '--num-iterations', default=10, type=int)
        self.add_passthru_arg('-l', '--num-locations', default=10, type=int)
        self.add_passthru_arg('-m', '--num-migrations', default=2, type=int)


    def get_finesses(self, population):
        fitnesses = []
        for idx in population:
            # fitness = evalDistance(pop, self.distance_matrix)
            fitness = self.calcDistance(self.locations[idx])
            fitnesses.append(fitness)
        
        fitnesses_inverse = 1/np.array(fitnesses)
        return fitnesses_inverse/np.sum(fitnesses_inverse), min(fitnesses)


    # TODO take this part out
    def initialize_population(self, _, values):
        num_loc = self.options.num_locations
        pop_size = self.options.population_size

        for island in range(4):
            population = np.empty(shape=(pop_size, num_loc), dtype=int)
            for i in range(pop_size):
                population[i] = np.random.permutation(num_loc)
            yield "island-{}".format(island+1), (population.tolist(), [])


    def mapper_init(self):
        from support.fitness import calcDistance
        from support.DNA import crossover, mutation

        with open(self.options.locations, 'r') as f:
            self.locations = np.array(json.load(f))

        self.calcDistance = calcDistance
        self.crossover = crossover
        self.mutation = mutation


    def mapper(self, key, island_info):
        population = np.array(island_info[0])
        shortest_distances = island_info[1]
        
        fitness, min_dist = self.get_finesses(population)
        if len(shortest_distances) == 0:
            shortest_distances.append(min_dist)

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
                child = self.crossover(*couple)

                # MUTATION
                if np.random.rand() < 0.01:
                    child = self.mutation(child)

                new_population.append(child)

            # DETERMINE FITNESS
            population = np.array(new_population)
            fitness, min_dist = self.get_finesses(population)
            shortest_distances.append(min_dist)

        best_idx = np.argsort(-fitness)

        if self.options.step_num < self.options.num_migrations+1:
            migrant_ids = best_idx[:6] # (num islands-1) *  number of migrants per island (2)
            all_migrants = np.random.permutation(population[migrant_ids]).tolist()

            mask = np.ones(len(population), dtype=bool)
            mask[migrant_ids] = False

            rest_pop = population[mask].tolist()

            for i in range(4):
                other = "island-{}".format(i+1)
                if other != key:
                    yield other, (all_migrants[:2], [])
                    all_migrants = all_migrants[2:]
                
            yield key, (rest_pop, shortest_distances)

        else:
            best = population[best_idx[0]].tolist()
            yield "result", (best, shortest_distances)


    def reducer(self, key, values):
        if self.options.step_num < self.options.num_migrations+1:
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
        return [MRStep(reducer=self.initialize_population)
                    ] + [MRStep(mapper_init=self.mapper_init,
                        mapper=self.mapper,
                        reducer=self.reducer)
                    ]*(self.options.num_migrations+1)


if __name__ == '__main__':
    IPGA.run()