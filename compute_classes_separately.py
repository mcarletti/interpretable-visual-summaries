import numpy as np
import multiprocessing
from joblib import Parallel, delayed


def get_params():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples_to_process', type=int, default=5000)
    parser.add_argument('--n_samples_per_class', type=int, default=25)
    parser.add_argument('--permute_classes', action='store_true')
    args = parser.parse_args()
    return args


def run(class_id, n_samples):
    from subprocess import call
    params = '--target_id %d --max_images %d' % (class_id, n_samples)
    try:
        call(['./run.sh'] + params.split())
    except Exception as e:
        print(e)


if __name__ == '__main__':

    args = get_params()

    max_class_id = 1000 # ILSVRC
    n_classes = args.n_samples_to_process // args.n_samples_per_class

    class_ids = np.arange(max_class_id)

    if args.permute_classes:
        permute = np.random.permutation(max_class_id)
        class_ids = class_ids[permute]

    class_ids = class_ids[:n_classes]


    try:
        n_cpus = multiprocessing.cpu_count()
        _ = Parallel(n_jobs=6)(delayed(run)(cid, args.n_samples_per_class) for cid in class_ids)
    except Exception as e:
        print(e)
