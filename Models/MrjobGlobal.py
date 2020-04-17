from mrjob.job import MRJob, MRStep
from Base import GA, DNA
import numpy as np
import json

class MRJobGlobal(MRJob, GA):
    FILES = ['../data/locations.json', '../Models/Base.py']

    def configure_args(self):
        super(MRJobGlobal, self).configure_args()
        self.add_passthru_arg('-p', '--population-size', default=10, type=int)
        self.add_passthru_arg('-n', '--num-iterations', default=10, type=int)
        self.add_passthru_arg('-l', '--num-locations', default=10, type=int)

        self.add_passthru_arg('-e', '--elite-fraction', default=0.1, type=float)
        self.add_passthru_arg('-m', '--mutation-rate', default=0.01, type=float)

    def mapper_init(self):
        self._get_locations_from_file('locations.json')

    def mapper_first(self, _, num):
        idx = np.random.permutation(self.options.num_locations)
        distance = DNA.fitness_func(self.locations[idx])
        yield "route", (idx.tolist(), distance)

    def mapper(self, key, values):
        if key == "result":
            yield "result", values

        if key == "elite":
            yield "route", values

        if key == "couples":
            children = DNA.crossoverAndMutation(*values, self.options.mutation_rate)

            for child in children:
                distance = DNA.fitness_func(self.locations[child])
                yield "route", (child.tolist(), distance)


    def reducer(self, key, values):
        if key == "result":
            for result in values:
                yield "result", result

        if key == "route":
            elite_size, num_parents = self._get_num_elites_and_parents()
            idx, distances = list(zip(*map(list, values)))


            if elite_size > 0:
                best_idxs = np.argsort(distances)[:elite_size]
                best_idx = best_idxs[0]
            else:
                best_idxs = []
                best_idx = np.argmin(distances)
            
            yield "result", (self.options.step_num, idx[best_idx], distances[best_idx])
            

            if self.options.step_num < self.options.num_iterations:
                for elite_idx in best_idxs:
                    yield "elite", (idx[elite_idx], distances[elite_idx])

                fitness = 1/np.array(distances)
                idx = np.array(idx)
                
                # SELECTION
                couple_idx = self._get_couple_idx(num_parents, fitness/np.sum(fitness))
            
                for couple in idx[couple_idx]:
                    yield "couples", couple.tolist()
            

    def reducer_last(self, key, values):
        for iter_num, dna, distance in sorted(values, key=lambda x: x[0]):
            yield dna, distance


    def steps(self):
        return [MRStep(
                    mapper_init=self.mapper_init,
                    mapper=self.mapper_first,
                    reducer=self.reducer)        
                ] + [MRStep(
                    mapper_init=self.mapper_init,
                    mapper=self.mapper,
                    reducer=self.reducer)        
                ]*self.options.num_iterations + [MRStep(
                        reducer=self.reducer_last)]


if __name__ == '__main__':
    MRJobGlobal.run()

