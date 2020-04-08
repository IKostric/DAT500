from __future__ import print_function

import sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt

from Timer import Timer

from Models.SGA import SGA
from Models.GlobalPGA import GPGA
from Models.IslandPGA import IPGA


class GA():
    def __init__(self):
        self._parseargs()


    def run(self):
        # init model
        self._select_model()

        # get initial population
        initial_population = self._get_initial_population()
        print("Running '{}' model.".format(self.options.model_type), file=sys.stdout)

        with Timer() as t:
            if self.options.model_type == "sequential":
                # run sequential ga
                self._run_sequential(initial_population)
            else:
                # run parallel ga
                self._run_mrjob()

        print('Job finished in {} seconds'.format(t.interval))
        print('Shortest distance is', self.model.best_fitnesses[-1])

        plt.figure()
        plt.plot(self.model.best_fitnesses)
        plt.show(block=False)

        plt.figure()
        self.model._plot(self.model.best)
        plt.show(block=False)

    def _get_initial_population(self):
        with open(self.options.locations, 'r') as f:
            loc = json.load(f)

        locations = np.array(loc['locations'])
        num_locations = locations.shape[-1]
        initial_population = []
        for _ in range(self.options.population_size):
            dna = np.random.permutation(num_locations).tolist()
            initial_population.append(dna)

        return np.array(initial_population)

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

    def _run_hadoop(self):
        with self.model.make_runner() as runner:
                runner.run()

                for key, value in self.model.parse_output(runner.cat_output()):
                    print(key, value)


    def _run_sequential(self, initial_population):
        self.model.run(initial_population)


    def _parseargs(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('-t', '--model-type', default='sequential')

        parser.add_argument('-l', '--locations', default='data/locations.json')
        parser.add_argument('-p', '--population-size', default=10, type=int)
        parser.add_argument('-n', '--num-iterations', default=10, type=int)

        self.options, self.args = parser.parse_known_args()

        # propagate to mrjob
        self.args += ['--num-iterations', self.options.num_iterations]
        self.args += ['--population-size', self.options.population_size]
        self.args += ['--locations', self.options.locations]
        self.args.insert(0, "data/input.txt")


if __name__ == '__main__':
    ga = GA()
    ga.run()
    import os
    print(os.getcwd())

    