from __future__ import print_function


import sys
from mrjob.job import MRJob, MRStep
import numpy as np
from target import evalOneMax
from datetime import datetime

ISLANDS = 3
POP = 6
DNAS = np.random.randint(2, size=(POP, 10))

class SimpleGA(MRJob):
    def mapper_init(self):
        for dna in DNAS:
            score = evalOneMax(dna)
            yield _, (dna.tolist(), score)   

    def mapper(self, key, val):
        pass
        # yield key, val

    def combiner(self, _, values):
        dnas, vals = map(list, zip(*values))
        sort_idx = np.argsort(vals)
        num = -int(POP/ISLANDS)
        dnas = np.array(dnas)[sort_idx][num:]
        vals = np.array(vals)[sort_idx][num:]
        for dna,v in zip(dnas, vals):
            yield _, (dna.tolist(), float(v))

    def reducer(self, key, values):
        for k, v in values:
            yield k,v

    def steps(self):
        return [MRStep(mapper_init=self.mapper_init),
                MRStep(mapper=self.mapper,
                   combiner=self.combiner,
                   reducer=self.reducer),
        ]

# def configure_args(self):
#     super(MRYourJob, self).configure_args()
#     self.add_passthru_arg(
#             '-n', '--num-iterations', default=1, type=int,
#             help='Number of times to run the job')

# def steps(self):
#     return [MRStep(mapper=self.mapper1, reducer=self.reducer1),
#                 MRStep(mapper=self.mapper2)] * self.options.num_iterations

if __name__ == '__main__':
    SimpleGA.run()