import os
import argparse

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
# The above code was critical due to a bizarre issue with multiprocessing
# that did not exist on my local machine, only the AWS and GCP CPUs.
# Without this code, each process involving numpy or scipy eigenvalue
# methods would consume all the CPUs available without any speedup.
import scripts.loop_graphs  # noqa: E402
import scripts.cyclic_covers  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--script-name", type=str)
parser.add_argument("-d", "--base_degree", type=int)
parser.add_argument("-s", "--base-sizes", type=int, nargs="+")
parser.add_argument("-cd", "--cover-degs", type=int, nargs="+")
parser.add_argument("-n", "--number", type=int)
parser.add_argument("-nc", "--number-covers", type=int)
parser.add_argument("--simple", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("-p", "--num_cpus", type=int)
args = parser.parse_args()

if args.num_cpus is None:
    import multiprocessing

    num_cpus = max(multiprocessing.cpu_count() - 1, 1)
else:
    num_cpus = args.num_cpus


if __name__ == "__main__":
    deg = args.base_degree
    for size in args.base_sizes:
        if args.script_name == "loop_graphs":
            scripts.loop_graphs.script_main(
                size=size,
                deg=args.base_degree,
                number=args.number,
                simple=args.simple,
                num_cpus=num_cpus,
            )
        elif args.script_name == "cyclic_covers":
            for cover_deg in args.cover_degs:
                scripts.cyclic_covers.script_main(
                    size=size,
                    deg=args.base_degree,
                    cover_deg=cover_deg,
                    number=args.number,
                    number_covers=args.number_covers,
                    num_cpus=num_cpus,
                )
        elif args.script_name == "quaternion_rep":
            scripts.quaternion_rep.script_main(
                size=size,
                deg=args.base_degree,
                number=args.number,
                num_cpus=num_cpus,
            )
        else:
            raise ValueError(
                "args.script_name must be one of "
                "quaternion_rep, loop_graphs, cyclic_covers"
            )
