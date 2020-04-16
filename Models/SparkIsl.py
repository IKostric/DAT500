# from mrjob.job import MRJob, MRStep

# from mrjob.protocol import TextProtocol
#%%
from Base import GA
from Sequential import SGA
import numpy as np
import json

class options():
    num_iterations = 10
    population_size = 20
    num_locations = 10

    mutation_rate = 0.01
    elite_fraction = 0.1

    migrant_fraction = 0.3
    num_migrations = 1
    num_islands = 4

class SparkIsland(GA):
    FILES = ['../data/locations.json', '../Models/Base.py', '../Models/Sequential.py']

    # OUTPUT_PROTOCOL = TextProtocol

    def __init__(self, options):
        self.options = options


    def spark(self):
        # import findspark
        # findspark.init()
        import pyspark
        sc = pyspark.SparkContext(appName="TSPIsland")
        self._get_locations_from_file('locations.json')

        #constants
        num_islands = self.options.num_islands
        locations = self.locations

        number_of_migrants = int(((options.population_size*options.migrant_fraction)
                                    /(num_islands-1))) # per island
        tot_number_of_migrants = number_of_migrants * (num_islands-1)


        workers = sc.parallelize(range(num_islands), num_islands).map(lambda x: (x, SGA(options, locations))).cache()
        populations = [None]*num_islands
        #  = workers.map(lambda ga: ga.run(island=True)).collect()
        
        print("\n\ntest\n\n")

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


            # populations = workers.map(lambda ga: ga.population)
        print(populations)
        # output = workers.map(lambda ga: ga[1].run(population=populations[ga[0]], island=False))
        # output.saveAsTextFile(output_path)
        

        sc.stop()

if __name__ == '__main__':
       
    ga = SparkIsland(options)
    ga.spark()

# %%
