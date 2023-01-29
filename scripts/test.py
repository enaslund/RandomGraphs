import numpy as np
import multiprocessing
import time
import sys
import random_graphs
import os


number_outer = 10
number_inner = 100
base_size = 100
cover_deg = 5


def async_func():
    # Critical line of code so the results of multiprocessing are
    # all identical
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    base_graph = random_graphs.random_simple_graph(size=base_size, deg=4)
    return random_graphs.generate_new_extremal_eigs(
        **{
            "base_graph": base_graph,
            "cover_deg": cover_deg,
            "permutation_func": random_graphs.permutations.abelian_cycle,
            "number": number_outer,
            "trivial_eig": 4,
            "eig_type": "max_positive",
        }
    )


def func_to_call():

    filename = sys.argv[1]

    start = time.time()
    results = []

    cpu_count = multiprocessing.cpu_count()
    print(f"CPU Count: {cpu_count}")
    pool = multiprocessing.Pool(6)

    for i in range(0, number_inner):
        result = pool.apply_async(async_func)
        results.append(result)

    eigs = []
    for x in results:
        eigs.extend(x.get())

    np.savetxt(filename + ".txt", eigs)
    print(time.time() - start)
