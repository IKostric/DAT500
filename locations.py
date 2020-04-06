import numpy as np
import sys, pickle, json


class Locations():
    def generate_random_locations(self, num_loc=10, dim=2):
        self.loc = np.random.random(size=(dim, num_loc))
        return self.loc

    def calculate_distances(self):
        distance_matrix = self.loc[:,None] - self.loc[:,:,None]
        distance_matrix_squared = np.sum(distance_matrix**2, axis=0)
        self.loc_matrix = np.sqrt(distance_matrix_squared)
        return self.loc_matrix

    def save_locations_to_file(self, filename='data/locations'):
        data_pickle = {
            'locations': self.loc, 
            'distances': self.loc_matrix
        }

        data_json = {
            'locations': self.loc.tolist(), 
            'distances': self.loc_matrix.tolist()
        }
        with open(filename+'.pickle', 'wb') as f:
            pickle.dump(data_pickle, f, pickle.HIGHEST_PROTOCOL)

        with open(filename+'.json', 'w') as f:
            json.dump(data_json, f)


class Location():
    def __init__(self, pos):
        self.pos = np.array(pos)

    def distance(self, other):
        return np.sqrt(np.sum((self.pos - other.pos)**2))


if __name__=='__main__':
    num_loc, dim = sys.argv[1:]
    loc = Locations()
    loc.generate_random_locations(int(num_loc), int(dim))
    loc.calculate_distances()
    loc.save_locations_to_file()

    print(loc.loc, '\n')
    print(loc.loc_matrix)