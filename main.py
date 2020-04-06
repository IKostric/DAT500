from __future__ import print_function
import sys 

# from mrjob.job import MRStep
from Timer import Timer
from Driver import GA
import sys

class RunGA():
    def __init__(self, args):
        self.args = args
        self.model_type = self.get_from_options('--type', default='sequential')
        self.n = int(self.get_from_options('--n', default=1))

    def start(self):
        ga = GA(self.model_type, args=self.args)
        with Timer() as t:
            ga.run()
            
        print("Elapsed time: {}".format(t.interval), file=sys.stdout)

    def get_from_options(self, param, default=None):
        value = default

        for arg in args:
            if (param in arg) and ('=' in arg):
                value = arg.split('=')[1]
                args.remove(arg)

        return value


if __name__ == '__main__':
    args = sys.argv[1:]
    print("Arguments: {}".format(args), file=sys.stdout)

    runner = RunGA(args)
    runner.start()

    