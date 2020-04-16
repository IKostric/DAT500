#%%
from __future__ import print_function

import sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt

from Timer import Timer
import Models

#%%
class Driver():
    def __init__(self, options=None):
        if options == None:
            self._parseargs()
        else:
            self.options = options
            self._add_passthru_args()
        self.model = None


    def run(self):
        # init model
        if self.model == None:
            self.select_model()
        print("Running '{}' model.".format(self.options.model_type))

        with Timer() as t:
            if self.options.model_type == "sequential":
                # run sequential ga
                result = self._run_sequential()
            else:
                # run parallel ga
                result = self._run_mrjob()

        print('Job finished in {} seconds'.format(t.interval))
        print('Shortest distance is {}'.format(result[1][-1]))

        self.plot(result)

    def select_model(self):
        model_type = self.options.model_type 

        if model_type == "sequential":
            self.model = Models.SGA(self.options)
        elif model_type == "global":
            self.model = Models.MRJobGlobal(self.args)
            self.prepare_input_file()
        elif model_type == "island":
            self.model = Models.MRJobIsland(self.args)
            self.prepare_input_file()
        elif model_type == "spark-g":
            self.model = Models.SparkGlobal(self.args)
        elif model_type == "spark-i":
            self.model = Models.SparkIsland(self.args)
        else:
            print(model_type)
            raise Exception("Model choices are: 'sequential', 'global' and 'island'")

    def prepare_input_file(self):
        num_lines = self.options.num_locations
        if self.options.model_type == 'island':
            num_lines = 4 # variable??

        lines = np.arange(num_lines, dtype=int)
        np.savetxt('data/input.txt', lines, fmt='%d')

    def plot(self, result):
        dna, history = result
        self._get_locations()

        self.plot_route(dna)
        plt.show(block=False)
        self.plot_trend(history)
        plt.show()

    def plot_route(self, dna):
        # TODO title, legend osv.
        loc = self.locations.T[:,dna]
        dna = np.pad(dna, (0, 1), 'wrap')
        plt.figure()
        plt.scatter(*loc)
        plt.plot(*loc)


    def plot_trend(self, arr):
        # TODO title, legend osv.
        plt.figure()
        plt.plot(arr)


    def _run_mrjob(self):
        with self.model.make_runner() as runner:
            runner.run()

            if self.options.model_type == "global":
                distances = []
                for idx, dist in self.model.parse_output(runner.cat_output()):
                    distances.append(dist)
            elif self.options.model_type == "island":
                for idx, distances in self.model.parse_output(runner.cat_output()):
                    pass
            else:
                distances = []
                for res, empty in self.model.parse_output(runner.cat_output()):
                    print(res.rstrip())
                    # distances.append(eval(res.rstrip()))

                distances = sorted(distances, key=lambda t: t[0])
                distances = [v for k,v in distances]
                print(distances)
                    # print("\n\n teset ", idx, dist.rstrip())
                
        return 1, np.array([1,2,3])#np.array(idx, dtype=int), np.array(distances)


    def _run_sequential(self):
        return self.model.run()

    def _get_locations(self):
        with open('data/locations.json', 'r') as f:
            self.locations = np.array(json.load(f))
       
    def _parseargs(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('-t', '--model-type', default='sequential')
        parser.add_argument('--no-plot')

        parser.add_argument('-p', '--population-size', default=10, type=int)
        parser.add_argument('-n', '--num-iterations', default=10, type=int)
        parser.add_argument('-l', '--num-locations', default=10, type=int)
        parser.add_argument('-e', '--elite_fraction', default=0.2, type=float)
        parser.add_argument('-m', '--mutation_rate', default=0.01, type=float)

        self.options, self.args = parser.parse_known_args()
        self._add_passthru_args(self.args)

    def _add_passthru_args(self, args=[]):
        # propagate to mrjob
        args += ['--num-locations', str(self.options.num_locations)]
        args += ['--num-iterations', str(self.options.num_iterations)]
        args += ['--population-size', str(self.options.population_size)]
        args.insert(0, "data/input.txt")

        self.args = args

#%%
if __name__ == '__main__':
    class options():
        model_type = 'sequential'
        num_iterations = 10
        population_size = 100
        num_locations = 100

        mutation_rate = 0.01
        elite_fraction = 0.1
        num_migrations = 0.1

    algorithm = Driver()
    # ga.options.num_locations = 100
    # ga.options.population_size = 100
    # ga.options.num_iterations = 1000
    algorithm.run()
    # import os
    # print(os.getcwd())


# %%
