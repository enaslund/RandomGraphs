import numpy as np
import multiprocessing
import time
import random_graphs
import os


def async_func(size, deg, number, simple):
    # Critical line of code so the results of multiprocessing are
    # all identical
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    return random_graphs.generate_loop_extremal_eigs(
        size=size, deg=deg, number=number, eig_type="max_positive", simple=True
    )


def script_main(size, deg, number, simple, num_cpus):
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
            kwds={"size": size, "deg": deg, "number": inner_number, "simple": simple},
        )
        results.append(result)

    eigs = []
    for x in results:
        eigs.extend(x.get())

    file_prefix = "simple" if simple else "basic"
    filename = file_prefix + f"loop_V{size}_N{number}_deg{deg}.npy"
    np.save(filename, np.array(eigs))
    print(f"Filename {filename} time: {np.round(time.time() - start,2)}s")
