from __future__ import print_function
import sys 

# from mrjob.job import MRStep
from datetime import datetime
from Driver import PGA
import sys

args = {

}


def timeit(n=1):
    elapsed_time = 0
    for _ in range(n):
        ga = PGA(args=sys.argv[1:])
        start_time = datetime.now()
        ga.run()
        end_time = datetime.now()
        elapsed_time += (end_time - start_time).total_seconds()
    return elapsed_time/n


if __name__ == '__main__':
    elapsed_time = timeit(1)
    print("Elapsed time: {}".format(elapsed_time), file=sys.stdout)