#%%
from __future__ import print_function

import sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt

from Timer import Timer
import Models

class Driver():
    def __init__(self, options=None):
        if options == None:
            self._parseargs()
        else:
            self.options = options
            self._add_passthru_args()
        self.model = None


    def run(self):
        num_repetitions = 3

        # init model
        if self.model == None:
            self.select_model()
        print("Running '{}' model.".format(self.options.model_type))

        sh = []
        plot_res = []
        times = []
        
        for i in range(num_repetitions):
            with Timer() as t:

                if self.options.model_type == "sequential":
                    # run sequential ga
                    result = self._run_sequential()
                else:
                    # run parallel ga
                    result = self._run_mrjob()
                
                sh.append(result[1][-1])
                plot_res.append(result[1])
            times.append(t.interval)

        plot_res = np.array(plot_res)
        # print(np.mean(plot_res, axis=0).tolist())
        with open('paths.txt', 'a+') as f:
            print(np.mean(plot_res, axis=0).tolist(), file=f)

        print(np.mean(times), np.std(times))
        print(sh, '\n')
        # if self.options.plot:
        #     self.plot(result)
        
        print(np.mean(sh), np.std(sh))

    def select_model(self):
        model_type = self.options.model_type 

        if model_type == "sequential":
            self.model = Models.SGA(self.options)
        elif model_type == "global":
            self.model = Models.MRJobGlobal(self.args)
            self.prepare_input_file()
        elif model_type == "island":
            self.args += ['--num-migrations', str(self.options.num_migrations)]
            self.args += ['--num-islands', str(self.options.num_islands)]
            self.model = Models.MRJobIsland(self.args)
            self.prepare_input_file()
        elif model_type == "global-s":
            self.model = Models.SparkGlobal(self.args)
        elif model_type == "island-s":
            self.model = Models.SparkIsland(self.args)
        else:
            print(model_type)
            raise Exception("Model choices are: 'sequential', 'global' and 'island'")

    def prepare_input_file(self):
        num_lines = self.options.population_size
        if self.options.model_type == 'island':
            num_lines = self.options.num_islands 

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
        dna = np.pad(dna, (0, 1), 'wrap')
        loc = self.locations.T[:,dna]
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
                for idx, dist in self.model.parse_output(runner.cat_output()):
                    distances = dist
                    
            elif self.options.model_type == "island-s":
                res = [eval(res.rstrip()) for res, empty in self.model.parse_output(runner.cat_output())]
                res = sorted(res, key=lambda r: r[1][-1])
                # res = list(zip(*map(list, res)))
                idx, distances = res[0]
                
            elif self.options.model_type == "global-s":
                res = [eval(res.rstrip()) for res, empty in self.model.parse_output(runner.cat_output())]
                res = sorted(res, key=lambda t: t[0])
                i, idx, dist = res[0]
                distances = [dist]

            else:
                return None, [0]
                
        return np.array(idx, dtype=int), np.array(distances)


    def _run_sequential(self):
        return self.model.run()

    def _get_locations(self):
        with open('data/locations.json', 'r') as f:
            self.locations = np.array(json.load(f))
       
    def _parseargs(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('-t', '--model-type', default='sequential')
        parser.add_argument('--plot', action='store_true')

        parser.add_argument('-p', '--population-size', default=10, type=int)
        parser.add_argument('-n', '--num-iterations', default=10, type=int)
        parser.add_argument('-l', '--num-locations', default=10, type=int)
        parser.add_argument('-e', '--elite-fraction', default=0.2, type=float)
        parser.add_argument('-m', '--mutation-rate', default=0.01, type=float)

        parser.add_argument('--num-islands', default=3, type=int)
        parser.add_argument('--num-migrations', default=4, type=int)

        self.options, self.args = parser.parse_known_args()
        self._add_passthru_args(self.args)

    def _add_passthru_args(self, args=[]):
        # propagate to mrjob
        args += ['--num-locations', str(self.options.num_locations)]
        args += ['--num-iterations', str(self.options.num_iterations)]
        args += ['--population-size', str(self.options.population_size)]
        args += ['--elite-fraction', str(self.options.elite_fraction)]
        args += ['--mutation-rate', str(self.options.mutation_rate)]
        args.insert(0, "data/input.txt")

        self.args = args


if __name__ == '__main__':
    class options():
        model_type = 'global'
        num_iterations = 1000
        population_size = 900
        num_locations = 1000

        num_islands = 6
        num_migrations = 4

        mutation_rate = 0.01
        elite_fraction = 0.2
        plot = True

    algorithm = Driver(options)
    # ga.options.num_locations = 100
    # ga.options.population_size = 100
    # ga.options.num_iterations = 1000
    algorithm.run()
    # import os
    # print(os.getcwd())


# %%
