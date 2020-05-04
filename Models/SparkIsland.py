from mrjob.job import MRJob, MRStep
from mrjob.protocol import TextProtocol
from Base import GA
from Sequential import SGA
import numpy as np
import json


class SparkIsland(MRJob, GA):
    FILES = ['../Models/Base.py', '../Models/Sequential.py']

    OUTPUT_PROTOCOL = TextProtocol

    def configure_args(self):
        super(SparkIsland, self).configure_args()
        self.add_passthru_arg('-p', '--population-size', default=10, type=int)
        self.add_passthru_arg('-n', '--num-iterations', default=10, type=int)
        self.add_passthru_arg('-l', '--num-locations', default=10, type=int)

        self.add_passthru_arg('-e', '--elite-fraction', default=0.1, type=float)
        self.add_passthru_arg('-m', '--mutation-rate', default=0.01, type=float)

        self.add_passthru_arg('--num-islands', default=3, type=int)
        self.add_passthru_arg('--num-migrations', default=2, type=int)
        self.add_passthru_arg('--migrant-fraction', default=0.3, type=float)


    def spark(self, input_path, output_path):
        # import findspark
        # findspark.init()
        import pyspark
        # pyspark.SparkContext.setSystemProperty('spark.yarn.am.cores', '1')
        # pyspark.SparkContext.setSystemProperty('spark.yarn.am.memory', '1g')
        pyspark.SparkContext.setSystemProperty('spark.driver.memory', '4g')
        pyspark.SparkContext.setSystemProperty('spark.executor.instances', '3')
        pyspark.SparkContext.setSystemProperty('spark.executor.cores', '2')
        pyspark.SparkContext.setSystemProperty('spark.executor.memory', '6g')
        sc = pyspark.SparkContext(appName="TSPIsland")
        self._get_locations_from_file('data/locations.json')

        #constants
        num_islands = self.options.num_islands
        options = self.options
        locations = self.locations

        number_of_migrants = int(((options.population_size*options.migrant_fraction)
                                    /(num_islands-1))) # per island
        tot_number_of_migrants = number_of_migrants * (num_islands-1)


        workers = sc.parallelize(range(num_islands), num_islands).map(lambda x: (x, SGA(options, locations))).cache()
        populations = [None]*num_islands
        #  = workers.map(lambda ga: ga.run(island=True)).collect()

        for _ in range(self.options.num_migrations):
            sorted_populations = np.array(workers
                    .map(lambda ga: ga[1].run(population=populations[ga[0]], island=True))
                    .collect())

            populations = [[] for _ in range(num_islands)]
            for island in range(num_islands):
                island_pop = sorted_populations[island]
                migrants = np.random.permutation(island_pop[:tot_number_of_migrants]).tolist()
                rest = island_pop[tot_number_of_migrants:].tolist()
                populations[island] += rest

                for other in range(num_islands):
                    if island != other:
                        populations[other] += migrants[:number_of_migrants]
                        migrants = migrants[number_of_migrants:]

            
        last_run = (workers
                    .map(lambda ga: ga[1].run(population=populations[ga[0]], island=False))
                    .map(lambda row: (row[0].tolist(), row[1]))
                    .saveAsTextFile(output_path))

        sc.stop()

if __name__ == '__main__':
    SparkIsland.run()