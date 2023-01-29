import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


import numpy as np  # noqa: E402
import multiprocessing
import time
import sys
import covers
import eigenvalues
import random_permutations

number_outer = int(sys.argv[1])
number_inner = int(sys.argv[2])
base_size = int(sys.argv[3])
cover_deg = int(sys.argv[4])


def async_func():
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    base_graph = covers.random_simple_graph(size=base_size, deg=4)
    return eigenvalues.generate_new_extremal_eigs(
        **{
            "base_graph": base_graph,
            "cover_deg": cover_deg,
            "permutation_func": random_permutations.abelian_cycle,
            "number": number_outer,
            "trivial_eig": 4,
            "eig_type": "max_positive",
        }
    )


if __name__ == "__main__":

    filename = sys.argv[5]

    start = time.time()
    results = []

    if len(sys.argv) > 6:
        cpu_count = int(sys.argv[6])
    else:
        cpu_count = multiprocessing.cpu_count()
    print(f"CPU Count: {cpu_count}")
    pool = multiprocessing.Pool(cpu_count)

    for i in range(0, number_inner):
        result = pool.apply_async(async_func)
        results.append(result)

    eigs = []
    for x in results:
        eigs.extend(x.get())

    np.savetxt(filename + ".txt", eigs)
    print(time.time() - start)
