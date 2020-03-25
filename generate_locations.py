import numpy as np
import sys
import json

num_loc = 10
if len(sys.argv) > 1:
    num_loc = int(sys.argv[1])
loc = np.random.random(size=(num_loc,2))

with open('input.txt', 'w') as f:
    json.dump({'locations': loc.tolist()}, f)