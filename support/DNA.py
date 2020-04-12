import numpy as np

#%%
def crossover(dna1, dna2):
    dna1 = np.array(dna1)
    dna2 = np.array(dna2)

    start, end = np.sort(np.random.choice(len(dna1), 2, replace=False))
    
    section = dna1[start:end]
    leftover = np.setdiff1d(dna2, section, assume_unique=True)

    child = np.empty_like(dna1)
    child[:start] = leftover[:start]
    child[start:end] = section
    child[end:] = leftover[start:]

    return child

def mutation(dna):
    start, end = np.random.choice(len(dna), 2, replace=False)
    mutated = dna.copy()
    mutated[start:end] = np.flip(mutated[start:end])
    return mutated

#%%
if __name__ == '__main__':
    dna1 = np.array([8, 7, 0, 5, 4, 9, 1, 3, 6, 2])
    dna2 = np.array([7, 0, 3, 5, 8, 1, 9, 4, 6, 2])
    print(crossover(dna1, dna2))
    print(mutation(dna1))

# %%
