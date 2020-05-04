import numpy as np
import sys, pickle, json


class Locations():
    def generate_random_locations(self, num_loc=1000, dim=2):
        """ Helper function for generating locations.

        Keyword Arguments:
            num_loc {int} -- Number of locations (default: {1000})
            dim {int} -- Number of dimensions (default: {2})

        Returns:
            [np.array] -- 2D array containg locations
        """
        self.loc = np.random.random(size=(num_loc, dim))
        return self.loc


    def save_locations_to_file(self, filename='data/locations'):
        """ Save locations to file.

        Keyword Arguments:
            filename {str} -- Name of the file where to store locations
                                 (default: {'data/locations'})
        """

        with open(filename+'.json', 'w') as f:
            json.dump(self.loc.tolist(), f)



if __name__=='__main__':
    assert len(sys.argv[1:]) == 2, "Please specify both num_locations and dimension" 
    num_loc, dim = sys.argv[1:]
    loc = Locations()

    loc.generate_random_locations(int(num_loc), int(dim))
    loc.save_locations_to_file()

    print(loc.loc, '\n')
    