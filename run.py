import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
# The above code was critical due to a bizarre issue with multiprocessing
# that did not exist on my local machine, only the AWS and GCP CPUs.
# Without this code, each process involving numpy or scipy eigenvalue
# methods would consume all the CPUs available without any speedup.

import scripts.test  # noqa: E402

if __name__ == "__main__":
    scripts.test.func_to_call()
# test.test_func()
