#%%
from __future__ import print_function

import sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt

from Timer import Timer

from Models.SGA import SGA
from Models.GlobalPGA import GPGA
from Models.IslandPGA import IPGA

#%%
class GA():
    def __init__(self):
        self._parseargs()


    def run(self):
        # init model
        self._select_model()
        print("Running '{}' model.".format(self.options.model_type))

        with Timer() as t:
            if self.options.model_type == "sequential":
                # run sequential ga
                idx, shortest = self._run_sequential()
            else:
                # run parallel ga
                idx, shortest = self._run_mrjob()

        print('Job finished in {} seconds'.format(t.interval))
        print('Shortest distance is {}'.format(shortest[-1]))

        self._get_locations()

        self.plot_route(idx)
        plt.show(block=False)
        self.plot_trend(shortest)
        plt.show()

    def _select_model(self):
        model_type = self.options.model_type 

        if model_type == "sequential":
            self.model = SGA(self.options)
        elif model_type == "global":
            self.model = GPGA(self.args)
        elif model_type == "island":
            self.model = IPGA(self.args)
        else:
            raise Exception("Model choices are: 'sequential', 'global' and 'island'")

    def _run_mrjob(self):
        with self.model.make_runner() as runner:
            runner.run()

            if self.options.model_type == "global":
                distances = []
                for idx, dist in self.model.parse_output(runner.cat_output()): 
                    distances.append(dist)
            else:
                for idx, distances in self.model.parse_output(runner.cat_output()):
                    pass

        return np.array(idx, dtype=int), np.array(distances)


    def _run_sequential(self):
        self.model.run()
        return self.model.best, self.model.best_fitnesses

    def plot_route(self, dna):
        # TODO title, legend osv.
        loc = self.locations.T
        dna = np.pad(dna, (0, 1), 'wrap')
        plt.figure()
        plt.scatter(*loc)
        plt.plot(*loc[:,dna])

    def plot_trend(self, arr):
        # TODO title, legend osv.
        plt.figure()
        plt.plot(arr)

    def _get_locations(self):
        with open(self.options.locations, 'r') as f:
            self.locations = np.array(json.load(f))
       

    def _parseargs(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('-t', '--model-type', default='sequential')
        parser.add_argument('-d', '--locations', default='data/locations.json')

        parser.add_argument('-p', '--population-size', default=10, type=int)
        parser.add_argument('-n', '--num-iterations', default=10, type=int)
        parser.add_argument('-l', '--num-locations', default=10, type=int)

        self.options, self.args = parser.parse_known_args()

        # propagate to mrjob
        self.args += ['--num-locations', str(self.options.num_locations)]
        self.args += ['--num-iterations', str(self.options.num_iterations)]
        self.args += ['--population-size', str(self.options.population_size)]
        self.args += ['--locations', self.options.locations]
        self.args.insert(0, "data/input.txt")

#%%
if __name__ == '__main__':
    ga = GA()
    # ga.options.num_locations = 100
    # ga.options.population_size = 100
    # ga.options.num_iterations = 1000
    ga.run()
    # import os
    # print(os.getcwd())


# %%
