import numpy as np
import time
import sys
import covers
import eigenvalues
import random_permutations


if __name__ == "__main__":
    number_outer = int(sys.argv[1])
    number_inner = int(sys.argv[2])
    base_size = int(sys.argv[3])
    cover_deg = int(sys.argv[4])
    filename = sys.argv[5]

    start = time.time()

    eigs = []
    for i in range(0, number_inner):
        base_graph = covers.random_simple_graph(size=base_size, deg=4)

        eigs += eigenvalues.generate_new_extremal_eigs(
            **{
                "base_graph": base_graph,
                "cover_deg": cover_deg,
                "permutation_func": random_permutations.abelian_cycle,
                "number": number_outer,
                "trivial_eig": 4,
                "eig_type": "max_positive",
            },
        )

    eigs = np.array(eigs)
    np.savetxt(filename + ".txt", eigs)
    print(time.time() - start)
