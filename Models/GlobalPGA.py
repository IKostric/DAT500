from mrjob.job import MRJob, MRStep
import numpy as np
import pickle


class GPGA(MRJob):
    DIRS = ['../data', '../support']

    def configure_args(self):
        super(MRYourJob, self).configure_args()
        self.add_passthru_arg('-l', '--locations', default='data/locations.json')
        self.add_passthru_arg('-p', '--population-size', default=10, type=int)
        self.add_passthru_arg('-n', '--num-iterations', default=10, type=int)

    def mapper_init(self):
        from support.fitness import evalDistance
        # with open('data/locations.pickle', 'rb') as f:
        #     # The protocol version used is detected automatically, so we do not
        #     # have to specify it.
        #     data = pickle.load(f)
            
        for i in range(5):#len(data['distances'])):
            yield _, str(i) + ",1,2,3"


    def mapper(self, _, value):
        from support.fitness import evalDistance
        # nodes = list(map(int, value.split(',')))
        yield _, value

    def reducer(self, _, values):
        # from support.DNA import crossover, mutation
        # TODO single reducer (single key)
        # provide population for the next iteration
        for value in values:
            yield _, value

    def reducer_last(self, _, values):
        # TODO single reducer (single key)
        # provide population for the next iteration
        for value in values:
            yield _, value

    def steps(self):
        return [MRStep(mapper_init=self.mapper_init,
                        # mapper=self.mapper,
                        reducer=self.reducer)
                    ] + [
                MRStep(mapper=self.mapper,
                        reducer=self.reducer)        
                    ]*self.options.num_iterations + [
                MRStep(mapper=self.mapper,
                        reducer=self.reducer_last)
                ]


if __name__ == '__main__':
    GPGA.run()