from mrjob.job import MRJob, MRStep
import numpy as np
import json



class GPGA(MRJob):
    DIRS = ['../data', '../support']

    def configure_args(self):
        super(GPGA, self).configure_args()
        self.add_passthru_arg('-d', '--locations', default='data/locations.json')
        self.add_passthru_arg('-p', '--population-size', default=10, type=int)
        self.add_passthru_arg('-n', '--num-iterations', default=10, type=int)
        self.add_passthru_arg('-l', '--num-locations', default=10, type=int)


    # TODO take this part out
    def initialize_population(self, _, values):
        val = int(next(values))
        for _ in range(self.options.population_size):
            yield "route", np.random.permutation(self.options.num_locations).tolist()


    def mapper_init(self):
        from support.fitness import calcDistance
        from support.DNA import crossover, mutation

        with open(self.options.locations, 'r') as f:
            self.locations = np.array(json.load(f))

        self.calcDistance = calcDistance
        self.crossover = crossover
        self.mutation = mutation


    def mapper_first(self, key, idx):
        distance = self.calcDistance(self.locations[idx])
        yield "route", (idx, distance)


    def mapper(self, key, values):
        if key == "result":
            yield "result", values

        if key == "elite":
            yield "route", values

        if key == "couples":
            # CROSSOVER
            child = self.crossover(*np.array(values))

            # MUTATION
            if np.random.rand() < 0.01: # could be parameter
                child = self.mutation(child)

            distance = self.calcDistance(self.locations[child])
            yield "route", (child.tolist(), distance)


    def reducer(self, key, values):
        if key == "result":
            for result in values:
                yield "result", result

        if key == "route":
            pop_size = self.options.population_size
            idx, distances = list(zip(*map(list, values)))

            best_idx = np.argmin(distances)
            yield "result", (self.options.step_num, idx[best_idx], distances[best_idx])
            
            if self.options.step_num < self.options.num_iterations:
                yield "elite", (idx[best_idx], distances[best_idx])

                fitness = 1/np.array(distances)
                idx = np.array(idx)
                
                # SELECTION
                couple_idx = np.random.choice(pop_size, \
                        size=(pop_size-1, 2), p=fitness/np.sum(fitness))
            
                for couple in idx[couple_idx]:
                    yield "couples", couple.tolist()
            

    def reducer_last(self, key, values):
        for iter_num, dna, distance in sorted(values, key=lambda x: x[0]):
            yield dna, distance


    def steps(self):
        return [MRStep(reducer=self.initialize_population)        
                    ] + [MRStep(mapper_init=self.mapper_init,
                        mapper=self.mapper_first,
                        reducer=self.reducer)        
                    ] + [MRStep(mapper_init=self.mapper_init,
                        mapper=self.mapper,
                        reducer=self.reducer)        
                    ]*self.options.num_iterations + [MRStep(
                        reducer=self.reducer_last)
                        ]


if __name__ == '__main__':
    GPGA.run()