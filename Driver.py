from Models.GlobalPGA import GPGA
from Models.IslandPGA import IPGA

class SGA():
    pass

class PGA():
    def __init__(self, model="global", args=[]):
        if model == "global":
            self.model = GPGA(args)
        elif model == "island":
            self.model = IPGA(args)
        else:
            raise Exception("Model choices are: 'global' and 'island'")

    def run(self):
        self.model.run()
