from mrjob.job import MRJob, MRStep
from mrjob.protocol import TextProtocol
from Base import GA
from Sequential import SGA
import numpy as np
import json


class SparkIsland(MRJob, GA):
    FILES = ['../data/locations.json', '../Models/Base.py', '../Models/Sequential.py']

    OUTPUT_PROTOCOL = TextProtocol

    def configure_args(self):
        super(SparkIsland, self).configure_args()
        self.add_passthru_arg('-p', '--population-size', default=10, type=int)
        self.add_passthru_arg('-n', '--num-iterations', default=10, type=int)
        self.add_passthru_arg('-l', '--num-locations', default=10, type=int)

        self.add_passthru_arg('-e', '--elite_fraction', default=0.1, type=float)
        self.add_passthru_arg('-m', '--mutation_rate', default=0.01, type=float)

        self.add_passthru_arg('--num-islands', default=4, type=int)
        self.add_passthru_arg('--num-migrations', default=2, type=int)
        self.add_passthru_arg('--migrant-fraction', default=0.3, type=float)


    def spark(self, input_path, output_path):
        # import findspark
        # findspark.init()
        import pyspark
        sc = pyspark.SparkContext(appName="TSPIsland")
        self._get_locations_from_file('locations.json')

        #constants
        num_islands = self.options.num_islands
        options = self.options
        locations = self.locations


        workers = sc.parallelize(range(num_islands), num_islands).map(lambda x: SGA(options, locations)).cache()
        first_run = workers.map(lambda x: x.run())        

        for _ in range(self.options.num_migrations):
            continue
            

        first_run.saveAsTextFile(output_path)

        sc.stop()

if __name__ == '__main__':
    SparkIsland.run()