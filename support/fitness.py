def evalOneMax(dna):
    return int(sum(dna))

def evalDistance(dna, distance_matrix):
    total_distance = distance_matrix[dna[-1], dna[0]]
    for i in range(len(dna)-1):
        total_distance += distance_matrix[dna[i], dna[i+1]]

    return total_distance
