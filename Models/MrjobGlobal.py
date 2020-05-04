from mrjob.job import MRJob, MRStep
from Base import GA, DNA
import numpy as np

class MRJobGlobal(MRJob, GA):
    """ 
    Class for running Global model of GA in MapReduce.
    Multiple inheritance from MrJob and GA.

    Arguments:
        MRJob {class} -- Inherit from MRJob, has all necessary methods for 
                            running the algorithm on hadoop cluster.
        GA {class} -- Has helper methods that all genetic algorithms use.

    Returns:
        class -- Instance of MRJob with extra methods.

    """

    # All workers should have this files
    FILES = ['../data/locations.json', '../Models/Base.py']


    def configure_args(self):
        """ Configure passthrough arguments that all workers 
        need to be able to run.
        """

        super(MRJobGlobal, self).configure_args()
        self.add_passthru_arg('-p', '--population-size', type=int)
        self.add_passthru_arg('-n', '--num-iterations', type=int)
        self.add_passthru_arg('-l', '--num-locations', type=int)

        self.add_passthru_arg('-e', '--elite-fraction', type=float)
        self.add_passthru_arg('-m', '--mutation-rate', type=float)


    def mapper_init(self):
        """ Runs once for all mappers on a single worker node. We load 
            data from file here instead of every time mapper runs.
        """

        self._get_locations_from_file('locations.json')


    def initialize(self, key, values):
        """ First mapper. Generates one individual for every line in 'input.txt' 
        file.

        Arguments:
            key {str} -- Nothing
            values {int} -- line number

        Yields:
            key, value -- key="individual", value=(dna, fitness)
        """

        idx = np.random.permutation(self.options.num_locations)
        distance = DNA.fitness_func(self.locations[idx])
        yield "individual", (idx.tolist(), distance)


    def mapper(self, key, values):
        """ Mappers for job no. 2 - num_iterations.

        Arguments:
            key {str} -- possible keys: 
                            "result", "elite", "individual"
            values {tuple} -- varies depending on key

        Yields for keys:
            "result" -- propagate result
            "elite" -- no changes to dna, yield individual
            "couples" -- yield children after crossover and mutation
        """

        if key == "result":
            yield "result", values

        if key == "elite":
            yield "individual", values

        if key == "couples":
            children = DNA.crossoverAndMutation(*values, self.options.mutation_rate)

            for child in children:
                distance = DNA.fitness_func(self.locations[child])
                yield "individual", (child.tolist(), distance)


    def reducer(self, key, values):
        """ Reducer for all jobs. Different yields for diffrent keys.

        Arguments:
            key {str} -- possible keys: 
                            "result", "individual"
            values {tuple} -- varies depending on key

        Yields for keys:
            "result" -- propagate intermidiate result
            "individual" -- yield best individual under key "result"
                         -- yield couples after selection process
        """

        if key == "result":
            for result in values:
                yield "result", result

        if key == "individual":
            elite_size, num_parents = self._get_num_elites_and_parents()
            idx, distances = list(zip(*map(list, values)))

            if elite_size > 0:
                best_idxs = np.argsort(distances)[:elite_size]
                best_idx = best_idxs[0]
            else:
                best_idxs = []
                best_idx = np.argmin(distances)
            
            yield "result", (self.options.step_num, idx[best_idx], distances[best_idx])
            
            # skip if we are on last iteration,
            # In that case only "result" key is being output
            if self.options.step_num < self.options.num_iterations:
                for elite_idx in best_idxs:
                    yield "elite", (idx[elite_idx], distances[elite_idx])

                fitness = 1/np.array(distances)
                idx = np.array(idx)
                
                # SELECTION
                couple_idx = self._get_couple_idx(num_parents, fitness/np.sum(fitness))
                
                for couple in idx[couple_idx]:
                    yield "couples", couple.tolist()
            

    # def reducer_last(self, key, values):
    #     for iter_num, dna, distance in sorted(values, key=lambda x: x[0]):
    #         yield dna, distance


    def steps(self):
        """ Define list of steps to take. Number of steps depends on number 
            of iterations.

        Returns:
            list -- List of MRSteps
        """
        return [MRStep(
                    mapper_init=self.mapper_init,
                    mapper=self.initialize,
                    reducer=self.reducer)        
                ]+[MRStep(
                    mapper_init=self.mapper_init,
                    mapper=self.mapper,
                    reducer=self.reducer)        
                ]*(self.options.num_iterations)


if __name__ == '__main__':
    MRJobGlobal.run()

