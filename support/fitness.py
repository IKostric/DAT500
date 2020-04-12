#%%
import numpy as np

def evalOneMax(dna):
    return int(sum(dna))

def evalDistance(dna, distance_matrix):
    total_distance = distance_matrix[dna[-1], dna[0]]
    for i in range(len(dna)-1):
        total_distance += distance_matrix[dna[i], dna[i+1]]

    return total_distance

def calcDistance(dna):
    dist = np.roll(dna,-1, axis=0) - dna
    if len(dist.shape) == 1:
        total_distance = np.sum(abs(dist))
    else:
        dist_sqr = np.sum(dist**2, axis=1)
        total_distance = np.sum(np.sqrt(dist_sqr))
    # total_distance = calcD(dna[-1], dna[0])
    # for i in range(len(dna)-1):
    #     total_distance += calcD(dna[i], dna[i+1])

    return total_distance

def calcD(val1, val2):
    return np.sqrt(np.sum((val1-val2)**2))
# %%
if __name__ == '__main__':
    dna = np.array([[1,3],[2,5],[7,2],[1,4]])
    # dna = np.array([1,6,2,4,7])
    print(calcDistance(dna))

# %%
    import matplotlib.pyplot as plt
    plt.scatter(*dna.T)

# %%
