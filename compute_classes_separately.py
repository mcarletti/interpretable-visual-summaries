import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import re


def get_params():
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--nb_samples_to_process', type=int, default=5000)
    parser.add_argument('--max_images_per_class', type=int, default=50)
    #parser.add_argument('--permute_classes', action='store_true')
    args = parser.parse_args()
    return args


def run(class_id, class_name, n_samples):
    try:
        from subprocess import call
        params = '--target_id %d --class_name %s --max_images %d' % (class_id, class_name, n_samples)
        call(['./run.sh'] + params.split())
    except Exception as e:
        print(e)


if __name__ == '__main__':

    args = get_params()

    '''
    max_class_id = 1000 # ILSVRC
    n_classes = args.nb_samples_to_process // args.max_images_per_class

    class_ids = np.arange(max_class_id)

    if args.permute_classes:
        permute = np.random.permutation(max_class_id)
        class_ids = class_ids[permute]

    class_ids = class_ids[:n_classes]
    '''

    class_info = []
    rc = np.loadtxt(open('RepresentativeClasses.csv'), dtype=object, delimiter='\n')

    for x in rc:
        cid = int(re.findall('\d+', x)[0])

        cname = x.split(':')[1].split(',')[0]
        cname = cname.strip().replace('\'', '').replace(' ', '_')

        class_info.append((cid, cname))

    try:
        n_cpus = multiprocessing.cpu_count()
        _ = Parallel(n_jobs=6)(delayed(run)(cid, cname, args.max_images_per_class) for cid, cname in class_info)
    except Exception as e:
        print(e)
