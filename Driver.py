from Models.SGA import SGA
from Models.GlobalPGA import GPGA
from Models.IslandPGA import IPGA
import sys



class GA():
    def __init__(self, model_type="sequential", args=[]):
        self.model_type = model_type
        self.args = args

        if model_type == "sequential":
            self.model = SGA(args)
        elif model_type == "global":
            self.model = GPGA(["data/input.txt"] + args)
        elif model_type == "island":
            self.model = IPGA(["data/input.txt"] + args)
        else:
            raise Exception("Model choices are: 'sequential', 'global' and 'island'")

    def run(self):
        print("Running '{}' model.".format(self.model_type), file=sys.stdout)
        if self.model_type == "sequential":
            self.model.run()
        else:
            with self.model.make_runner() as runner:
                runner.run()

                for key, value in self.model.parse_output(runner.cat_output()):
                    print(key, value)
