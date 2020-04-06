from support.fitness import evalDistance


def getFileContents(filename):
    with open(filename, 'r') as f:
        contents = [line for line in f]

    return contents


class SGA():
    def __init__(self, args):
        self.args = args
        self.filename = args[0]
        self.n = 1

    def run(self):
        # GET INITIAL POPULATION
        initial_population = getFileContents(self.filename)
        print(initial_population)

        # DETERMINE FITNESS

        for _ in range(self.n):
            continue
            # SELECTION

            # CROSSOVER

            # MUTATION

            # DETERMINE FITNESS



