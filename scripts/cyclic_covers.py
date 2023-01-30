import numpy as np
import multiprocessing
import time
import random_graphs
import os


def async_func(
    base_size,
    cover_deg,
    deg,
    inner_number,
):
    # Critical line of code so the results of multiprocessing are
    # all identical
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    base_graph = random_graphs.random_simple_graph(size=base_size, deg=deg)
    return random_graphs.generate_new_extremal_eigs(
        **{
            "base_graph": base_graph,
            "cover_deg": cover_deg,
            "permutation_func": random_graphs.permutations.abelian_cycle,
            "number": inner_number,
            "trivial_eig": deg,
            "eig_type": "max_positive",
        }
    )


def script_main(size, deg, cover_deg, number, number_covers, num_cpus):
    start = time.time()
    results = []

    pool = multiprocessing.Pool(num_cpus)

    for i in range(0, number):
        result = pool.apply_async(
            async_func,
            kwds={
                "base_size": size,
                "cover_deg": cover_deg,
                "deg": deg,
                "inner_number": number_covers,
            },
        )
        results.append(result)

    eigs = []
    for x in results:
        eigs.extend(x.get())

    filename = (
        f"abeliancover_V{size}x{cover_deg}_N{number}x{number_covers}_bdeg{deg}.npy"
    )

    np.save(filename, np.array(eigs))
    print(f"Filename {filename} time: {np.round(time.time() - start,2)}s")
