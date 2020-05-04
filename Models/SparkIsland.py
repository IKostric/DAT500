from mrjob.job import MRJob, MRStep
from mrjob.protocol import TextProtocol
from Base import GA
from Sequential import SGA
import numpy as np
import json


class SparkIsland(MRJob, GA):
    """ Class for running the Island model of GA in Spark.
    Multiple inheritance from MrJob and GA.

    Arguments:
        MRJob {class} -- Inherit from MRJob, has all necessary methods for 
                            running the algorithm on hadoop cluster.
        GA {class} -- Has helper methods that all genetic algorithms use.

    Returns:
        class -- Instance of MRJob with extra methods. 

    """

    # All workers should have this files
    # FILES = ['../Models/Base.py', '../Models/Sequential.py']

    # Output protocol
    OUTPUT_PROTOCOL = TextProtocol

    def configure_args(self):
        """ Configure passthrough arguments that all workers 
        need to be able to run.
        """

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
        """ Spark method of MRJob class. Acts the same spark-submit with added 
        conveniance of uploading neccessary files to the cluster and setting up 
        input and output path. This method uses PySpark to run Island model of
        genetic algorithm in Spark.

        Arguments:
            input_path {str} -- input path, ignored in our case
            output_path {str} -- path where to save the output of the algorithm.

        """

        import pyspark
        # Set up driver and worker properties.
        pyspark.SparkContext.setSystemProperty('spark.driver.memory', '6g')
        pyspark.SparkContext.setSystemProperty('spark.executor.instances', '3')
        pyspark.SparkContext.setSystemProperty('spark.executor.cores', '1')
        pyspark.SparkContext.setSystemProperty('spark.executor.memory', '3g')

        # Initialize spark context
        sc = pyspark.SparkContext(appName="TSPIsland")
        self._get_locations_from_file('data/locations.json')

        #constants
        num_islands = self.options.num_islands
        options = self.options
        locations = self.locations

        # get number of migrants
        number_of_migrants = int(((options.population_size*options.migrant_fraction)
                                    /(num_islands-1))) # per island
        tot_number_of_migrants = number_of_migrants * (num_islands-1)

        # parallelize SGA to every worker
        workers = sc.parallelize(range(num_islands), num_islands).map(lambda x: (x, SGA(options, locations))).cache()
        populations = [None]*num_islands

        # repeat SGA and migrations
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

        # Do final run after last migration occured
        last_run = (workers
                    .map(lambda ga: ga[1].run(population=populations[ga[0]], island=False))
                    .map(lambda row: (row[0].tolist(), row[1]))
                    .saveAsTextFile(output_path))

        sc.stop()

if __name__ == '__main__':
    SparkIsland.run()