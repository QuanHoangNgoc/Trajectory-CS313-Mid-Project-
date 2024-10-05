def apriori_algorithm(data, min_support):
    # Step 1: Generate 1-coordinate sets
    C1 = generate_initial_candidates(data)
    L1 = filter_candidates(C1, min_support)
    
    print(C1)
    print(L1)
    # Step 2: Iteratively generate k-coordinate sets
    k = 2
    Lk = L1
    while Lk:
        Ck = generate_candidates(Lk, k)
        Lk = filter_candidates(Ck, min_support)
        print(Ck)
        print(Lk) 
        k += 1
    
    return Lk

def generate_initial_candidates(data):
    # Generate initial 1-coordinate sets
    C1 = {}
    for trajectory in data:
        for coord in trajectory:
            if coord in C1:
                C1[coord] += 1
            else:
                C1[coord] = 1
    return C1

def filter_candidates(Ck, min_support):
    # Filter candidates based on minimum support
    Lk = {coord: count for coord, count in Ck.items() if count >= min_support}
    return Lk

def generate_candidates(Lk, k):
    # Generate k-coordinate candidate sets
    Ck = {}
    for coord1 in Lk:
        for coord2 in Lk:
            if coord1 != coord2:
                candidate = tuple(sorted(set(coord1) | set(coord2)))
                if len(candidate) == k:
                    Ck[candidate] = 0
    return Ck

# Example usage
data = [
    [(1,1), (1,2), (2,2), (2,3), (3,4), (3,5)],
    [(1,1), (1,2), (2,2), (2,3), (3,4), (3,5)],
    # Add more trajectories here
]
min_support = 1 
frequent_patterns = apriori_algorithm(data, min_support)
print(frequent_patterns)
