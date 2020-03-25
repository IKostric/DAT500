from mrjob.job import MRJob, MRStep
import numpy as np

class GPGA(MRJob):
    def mapper_init(self):
        pass   

    def mapper(self, _, val):
        yield val, 1

    def combiner(self, k, v):
        pass
        # yield k, sum(list(v))

    def reducer(self, k, v):
        yield k, sum(list(v))

    def steps(self):
        return [MRStep(#mapper_init=self.mapper_init,
                    mapper=self.mapper,
                   #combiner=self.combiner,
                   reducer=self.reducer),
        ]*10


if __name__ == '__main__':
    GPGA.run()