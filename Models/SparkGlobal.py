from mrjob.job import MRJob, MRStep
from mrjob.protocol import TextProtocol
from Base import GA, DNA
import numpy as np


class SparkGlobal(MRJob, GA):    
    """ Class for running Global model of GA in Spark.
    Multiple inheritance from MrJob and GA.

    Arguments:
        MRJob {class} -- Inherit from MRJob, has all necessary methods for 
                            running the algorithm on hadoop cluster.
        GA {class} -- Has helper methods that all genetic algorithms use.

    Returns:
        class -- Instance of MRJob with extra methods. 

    """

    # All workers should have this files
    FILES = ['../Models/Base.py', '../data/locations.json']

    # Output protocol
    OUTPUT_PROTOCOL = TextProtocol

    def configure_args(self):
        """ Configure passthrough arguments that all workers 
        need to be able to run.
        """

        super(SparkGlobal, self).configure_args()
        self.pass_arg_through('--runner')

        self.add_passthru_arg('-p', '--population-size', default=10, type=int)
        self.add_passthru_arg('-n', '--num-iterations', default=10, type=int)
        self.add_passthru_arg('-l', '--num-locations', default=10, type=int)

        self.add_passthru_arg('-e', '--elite-fraction', default=0.2, type=float)
        self.add_passthru_arg('-m', '--mutation-rate', default=0.01, type=float)


    def spark(self, input_path, output_path):
        """ Spark method of MRJob class. Acts the same spark-submit with added 
        conveniance of uploading neccessary files to the cluster and setting up 
        input and output path. This method uses PySpark to run Global model of
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

        # instantiate sparkcontext
        sc = pyspark.SparkContext(appName ='TSPGlobal')

        # load data
        filename = 'locations.json'
        if self.options.runner == 'spark':
            filename = 'data/' + filename
        self._get_locations_from_file(filename)

        # set constants
        num_iterations = self.options.num_iterations
        mrate = self.options.mutation_rate
        elite_size, num_parents = self._get_num_elites_and_parents()

        # Broadcasts information that every worker should have
        ga = sc.broadcast(DNA)
        locations = sc.broadcast(self.locations)


        # spark helper functions
        def get_distance(dna):
            """ Calculate fitness for the individual.

            Arguments:
                dna {list} -- list of indices.

            Returns:
                tuple -- individual, distance, score
            """

            distance = ga.value.fitness_func(locations.value[dna])
            return dna, distance, 1/distance


        def breed(couple):
            """ Given couple of individuals return children after
            crossover and mutation

            Arguments:
                couple {couple} -- list of two parents

            Yields:
                list -- child
            """

            children = ga.value.crossoverAndMutation(*couple, mrate)
            for child in children:
                yield child


        def unwrap_mat(dist):
            """ Change the structure of the matrix after collecting
            entire population.


            Returns:
                tuple -- population, distances, fitness
            """
            population, distances, fitness = dist.T
            population = np.stack(population)
            fitness = fitness/np.sum(fitness)
            return population, distances, fitness

        best_fitnesses = np.empty(num_iterations+1)

        # initialize populations
        initial_population = self._get_initial_population()
        init_population = sc.parallelize(initial_population)

        # get fitnesses
        dist = np.array(init_population
                    .map(get_distance)
                    .collect())

        population, distances, fitness = unwrap_mat(dist)
        best_ids = np.argsort(distances)
        shortest = [distances[best_ids[0]]]

        # run iterative process
        for _ in range(num_iterations):
            
            couple_idx = self._get_couple_idx(num_parents, fitness.astype(np.float))

            elite = sc.parallelize(np.atleast_2d(population[best_ids[:elite_size]]))
            couples = sc.parallelize(population[couple_idx])
            children = couples\
                .flatMap(breed)
            
            # combine elites and newly created children
            new_population = children.union(elite)
            dist = np.array(new_population
                    .map(get_distance)
                    .collect())

            # get fitnesses
            population, distances, fitness = unwrap_mat(dist)
            best_ids = np.argsort(distances)
            shortest.append(distances[best_ids[0]])
            
        # save to file
        sc.parallelize([population[best_ids[0]].tolist(), shortest])\
            .saveAsTextFile(output_path)
        sc.stop()

if __name__ == '__main__':
    SparkGlobal.run()