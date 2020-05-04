#%%
from __future__ import print_function

import sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt

from Timer import Timer
import Models, json

class Driver():
    """ Entry point for all genetic algorithms.

        usage: 
        $ python Driver.py -t <model type> <options>
    """
    def __init__(self, options=None):
        if options == None:
            self._parseargs()
        else:
            self.options = options
            self._add_passthru_args()
        self.model = None


    def run(self, num_repetitions=1):
        """ Run selected model. Print average timings and results.

        Keyword Arguments:
            num_repetitions {int} -- Number of times to repeat the experiment (default: {1})
        """

        # init model
        if self.model == None:
            self.select_model()
        print("Running '{}' model.".format(self.options.model_type))

        
        sh = []
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
            times.append(t.interval)

        print("Finished \n", self.options)
        print('Job finished in {} +- {} seconds'.format(np.mean(times), np.std(times)))
        print('Shortest distance is {} +- {}'.format(np.mean(sh), np.std(sh)))
        print('times: ', times)
        print('\n')
        
        # plot if option set
        if self.options.plot:
            self.plot(result)
        

    def select_model(self):
        """ Select appropriate model given '-t' parameter.

        Raises:
            Exception: Non existent model.
        """
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
            raise Exception("Model choices are: 'sequential', 'global', 'island', 'global-s', 'island-s'")

    def prepare_input_file(self):
        """ Helper method for MapReduce. Creates input file called 'input.txt'.
            If model is global, number of lines is equal to population size, that
            way every mapper gets one individual from the population.

            If model is island, number of lines is equal to number of islands. In
            this case number of mappers is equal to number of islands.
        """
        num_lines = self.options.population_size
        if self.options.model_type == 'island':
            num_lines = self.options.num_islands 

        lines = np.arange(num_lines, dtype=int)
        np.savetxt('data/input.txt', lines, fmt='%d')

    def plot(self, result):
        """ Helper method for plotting.

        Arguments:
            result {tuple} -- First element is the best individual, second is the 
                                list of shortest paths in every iteration
        """
        dna, history = result
        self._get_locations()

        # plot route
        if self.locations.size == 2:
            self.plot_route(dna)
            plt.show(block=False)

        # plot convergence
        self.plot_trend(history)
        plt.show()

    def plot_route(self, dna):
        """ Helper method for plotting route.

        Arguments:
            dna {list} -- list of indices, order in which to travel between locations. 
        """
        dna = np.pad(dna, (0, 1), 'wrap')
        loc = self.locations.T[:,dna]
        plt.figure()
        plt.scatter(*loc)
        plt.plot(*loc)
        plt.title("Best route path after {} iterations".format(self.options.num_iterations))
        plt.xlabel("x")
        plt.ylabel("y")


    def plot_trend(self, arr):
        """ Helper method for plotting convergence.

        Arguments:
            arr {list} -- list of shortest routes after every iteration.
        """
        plt.figure()
        plt.plot(arr)
        plt.title("Convergence rate")
        plt.xlabel("Number of iterations")
        plt.ylabel("Distance")


    def _run_mrjob(self):
        """ Run MrJob programmaticaly and parse result based on the model.

        Returns:
            [tuple] -- tuple containing best individual and shortest path at every iteration.
        """
        with self.model.make_runner() as runner:
            runner.run()

            if self.options.model_type == "global":
                distances = []
                counter = -1
                for key, value in self.model.parse_output(runner.cat_output()):
                    iter_num, idxs, dist = value
                    if iter_num > counter:
                        counter = iter_num
                        distances.append(dist)
                        idx = idxs

            elif self.options.model_type == "island":
                for idx, dist in self.model.parse_output(runner.cat_output()):
                    distances = dist
                    
            elif self.options.model_type == "island-s":
                res = [eval(res.rstrip()) for res, empty in self.model.parse_output(runner.cat_output())]
                res = sorted(res, key=lambda r: r[1][-1])
                # res = list(zip(*map(list, res)))
                idx, distances = res[0]
                
            elif self.options.model_type == "global-s":
                res = next(runner.cat_output())
                idx = json.loads(res.rstrip())
                res = next(runner.cat_output())
                distances = json.loads(res.rstrip())

            else:
                return None, [0]
                
        return np.array(idx, dtype=int), np.array(distances)


    def _run_sequential(self):
        """ Run SGA, this is done localy so we don't need MrJob.

        Returns:
            [tuple] -- tuple containing best individual and shortest path at every iteration.
        """
        return self.model.run()

    def _get_locations(self):
        """ Load locations. This is mostly used for plotting. Models load their own instances 
            of locations.

            locations is a list of locations with dimensions (num_locations x dimension)
        """
        with open('data/locations.json', 'r') as f:
            self.locations = np.array(json.load(f))
       
    def _parseargs(self):
        """ Parse arguments from command line.
        """
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
        """ Add passthrough arguments. This is requred for MapReduce passthourgh args.


        Keyword Arguments:
            args {list} -- Already set arguments (default: {[]})
        """
        # propagate to mrjob
        args += ['--num-locations', str(self.options.num_locations)]
        args += ['--num-iterations', str(self.options.num_iterations)]
        args += ['--population-size', str(self.options.population_size)]
        args += ['--elite-fraction', str(self.options.elite_fraction)]
        args += ['--mutation-rate', str(self.options.mutation_rate)]
        args.insert(0, "data/input.txt")

        self.args = args


if __name__ == '__main__':
    """ Run the algorithm.
    """
    # Uncomment options class to speccify the options here instead of command line.
    # class options():
    #     model_type = 'global'
    #     num_iterations = 1000
    #     population_size = 900
    #     num_locations = 1000

    #     num_islands = 6
    #     num_migrations = 4

    #     mutation_rate = 0.01
    #     elite_fraction = 0.2

    if "options" in locals():
        algorithm = Driver(options)
    else:
        algorithm = Driver()
    algorithm.run()


