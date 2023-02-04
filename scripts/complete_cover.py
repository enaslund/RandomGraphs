import numpy as np
import multiprocessing
import time
import random_graphs
import os


def async_func(
    base_size,
    cover_deg,
    inner_number,
):
    # Critical line of code so the results of multiprocessing are
    # all identical
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    # This generates the complete graph on 4 vertices
    base_graph = np.matrix(np.ones((base_size, base_size)) - np.eye(base_size))
    return random_graphs.generate_new_extremal_eigs(
        **{
            "base_graph": base_graph,
            "cover_deg": cover_deg,
            "permutation_func": np.random.permutation,
            "number": inner_number,
            "trivial_eig": 3,
            "eig_type": "max_positive",
        }
    )


def script_main(base_size, cover_deg, number, num_cpus):
    start = time.time()
    results = []

    pool = multiprocessing.Pool(num_cpus)

    if number > 10 * num_cpus:
        nbatches = num_cpus * 10
        batch_size = int(number / nbatches)
        remaining = number - nbatches * batch_size
        inner_numbers = [batch_size] * nbatches + [remaining]
    else:
        inner_numbers = [1] * number

    for inner_number in inner_numbers:
        result = pool.apply_async(
            async_func,
            kwds={
                "base_size": base_size,
                "cover_deg": cover_deg,
                "inner_number": inner_number,
            },
        )
        results.append(result)

    eigs = []
    for x in results:
        eigs.extend(x.get())

    filename = f"complete_cover_V{cover_deg}x{base_size}_N{number}.npy"

    np.save(filename, np.array(eigs))
    print(f"Filename {filename} time: {np.round(time.time() - start,2)}s")
