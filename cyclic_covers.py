import numpy as np
import multiprocessing
import time
import sys
import covers
import eigenvalues
import random_permutations


if __name__ == "__main__":
    number = int(sys.argv[1])
    base_size = int(sys.argv[2])
    cover_deg = int(sys.argv[3])
    filename = sys.argv[4]

    start = time.time()
    results = []

    cpu_count = multiprocessing.cpu_count()
    print(f"CPU Count: {cpu_count}")
    pool = multiprocessing.Pool(cpu_count)

    for i in range(0, number):
        base_graph = covers.random_simple_graph(size=base_size, deg=4)
        result = pool.apply_async(
            eigenvalues.generate_new_extremal_eigs,
            kwds={
                "base_graph": base_graph,
                "cover_deg": cover_deg,
                "permutation_func": random_permutations.abelian_cycle,
                "number": 1,
                "trivial_eig": 4,
                "eig_type": "max_positive",
            },
        )
        results.append(result)

    eigs = []
    for x in results:
        eigs.extend(x.get())

    np.savetxt(filename + ".txt", eigs)
    print(time.time() - start)
