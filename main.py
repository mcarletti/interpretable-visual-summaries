import torch
import numpy as np
import cv2
import os
import time

import explain
import utils
from regularizers import l1_reg, tv_reg, less_reg, lasso_reg

from collections import namedtuple
from recordclass import recordclass # mutable version of namedtuple


def get_params():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname',    type=str, default='alexnet')
    parser.add_argument('--input_path',   type=str, default='examples/original')
    parser.add_argument('--dest_folder',  type=str, default='results/')
    parser.add_argument('--results_file', type=str, default='results/results.csv')
    parser.add_argument('--file_ext',     type=str, default='.jpg')
    parser.add_argument('--max_images',   type=int, default=None)
    parser.add_argument('--target_id',    type=int, default=None)
    parser.add_argument('--no_super_pixel',  action='store_true')
    parser.add_argument('--verbose',      action='store_true')
    args = parser.parse_args()
    return args


def load_mask(filename, target_shape=None, mode='uniform'):
    if filename is not None and os.path.exists(filename):
        mask_init = cv2.imread(filename, 1)
        mask_init = cv2.cvtColor(mask_init, cv2.COLOR_BGR2GRAY)
        mask_init = np.float32(mask_init) / 255
        mask_init = 1. - mask_init # revert the activations
    else:
        assert target_shape is not None
        assert mode in ['uniform', 'circular']
        # 0 = perturbate, 1 = do not perturbate
        mask_init = np.zeros(target_shape, dtype = np.float32)
        if mode == 'circular':
            radius = target_shape[0] // 2 # assuming square shape
            cx, cy = target_shape[0] // 2, target_shape[1] // 2
            x, y = np.ogrid[-cx:target_shape[0]-cx, -cy:target_shape[1]-cy]
            pixels = x*x + y*y > radius*radius
            mask_init[pixels] = 1.
    return mask_init


def compute_stats(target, output, blurred):
    drop = (target - output) / target
    p = (output - target) / (target - blurred)
    return drop, p


def run_evaluation(model_info, impath, out_folder=None, gpu_id=0, verbose=False):
    
    print('Processing', impath, 'on gpu', gpu_id)

    t_start = time.time()

    model, target_shape = model_info
    
    data = ''
    out_file = args.results_file + '.part' + str(gpu_id)
    if not os.path.exists(out_file):
        data = 'filename, predicted_class_id, target_prob, smooth_mask_prob, smooth_drop, smooth_blurred_prob, smooth_p, sharp_mask_prob, sharp_drop, sharp_blurred_prob, sharp_p, spx_mask_prob, spx_drop, spx_blurred_prob, spx_p\n'

    # load the BGR image, resize it to the right resolution
    original_img = cv2.imread(impath, 1)
    original_img = cv2.resize(original_img, target_shape)

    if out_folder is None:
        out_folder = args.dest_folder

    #--------------------------------------------------------------------------

    if verbose:
        print('*' * 12 + '\nComputing blurred heatmap')

    params = Params(
        learning_rate = 0.1,
        max_iterations = 300,
        tv_beta = 3,      # exponential tv factor
        l1_coeff = 0.01,  # reduces number of masked pixels
        tv_coeff = 1e-4,  # encourages compact and smooth heatmaps
        less_coeff = 0.,  # encourages similarity between predictions
        lasso_coeff = 0., # force masked pixels to binary values
        noise_sigma = 0., # sigma of additional perturbation
        noise_scale = 1., # scale factor of additional perturbation
        blur = True,      # blur upsampled mask during optimization
        target_shape = target_shape,
        target_id = args.target_id)

    # assume to mask all the pixels (all ones)
    # as described in the paper, the resolution of the initial mask
    # is lower than the processed image because we want to avoid
    # artifacts in the optimization process; this causes also a "natural"
    # mask which is blurred due to the upsampling
    resize_factor = 8
    mask_shape = (target_shape[0] // resize_factor, target_shape[1] // resize_factor)
    mask_init = load_mask(None, mask_shape, 'circular')

    results = explain.compute_heatmap(model, original_img, params, mask_init, HAS_CUDA, gpu_id)
    if results is None:
        print("Wrong classification! Skipping")
        return
    upsampled_mask, blurred_img_numpy, target_prob, output_prob, blurred_prob, history, class_id = results

    if verbose:
        print('Prediction drops to {:.6f}'.format(output_prob))
    out = os.path.join(out_folder, 'smooth')
    utils.save(original_img, blurred_img_numpy, upsampled_mask, dest_folder=out, save_orig=True)

    smooth_drop, smooth_p = compute_stats(target_prob, output_prob, blurred_prob)
    data += impath + ',' + str(class_id) + ',' + str(target_prob) + ',' + str(output_prob) + ',' + str(smooth_drop) + ',' + str(blurred_prob) + ',' + str(smooth_p)

    # save loss history
    tmp = 'total_loss, class_loss, l1_loss, tv_loss, lasso_loss, less_loss\n'
    for t in history:
        t = [str(v) + ',' for v in t]
        t = ''.join(t) + '\n'
        tmp += t
    with open(os.path.join(out_folder, 'smooth/loss_history.csv'), 'w') as fp:
        fp.write(tmp)

    #--------------------------------------------------------------------------

    if verbose:
        print('*' * 12 + '\nComputing sharp heatmap')

    params = Params(
        learning_rate = 0.1,
        max_iterations = 150,
        tv_beta = 3,
        l1_coeff = 0.01,
        tv_coeff = 1.,
        less_coeff = 0.,
        lasso_coeff = 2.,
        noise_sigma = 0.,
        noise_scale = 1.,
        blur = False,
        target_shape = target_shape,
        target_id = args.target_id)

    # we want to generate a sharp heatmap to simplify the region
    # explanation and object detection/segmentation
    # since working on a high resolution mask causes scattered
    # results, we use as initialization mask the low resolution
    # output from the previous optimization (reference paper)
    mask_path = os.path.join(out_folder, 'smooth/mask.png')
    mask_init = load_mask(mask_path)

    results = explain.compute_heatmap(model, original_img, params, mask_init, HAS_CUDA, gpu_id, verbose=False)
    upsampled_mask, blurred_img_numpy, target_prob, output_prob, blurred_prob, history, class_id = results

    if verbose:
        print('Prediction drops to {:.6f}'.format(output_prob))
    out = os.path.join(out_folder, 'sharp')
    utils.save(original_img, blurred_img_numpy, upsampled_mask, dest_folder=out, save_orig=False)

    sharp_drop, sharp_p = compute_stats(target_prob, output_prob, blurred_prob)
    data += ',' + str(output_prob) + ',' + str(sharp_drop) + ',' + str(blurred_prob) + ',' + str(sharp_p)

    # save loss history
    tmp = 'total_loss, class_loss, l1_loss, tv_loss, lasso_loss, less_loss\n'
    for t in history:
        t = [str(v) + ',' for v in t]
        t = ''.join(t) + '\n'
        tmp += t
    with open(os.path.join(out_folder, 'sharp/loss_history.csv'), 'w') as fp:
        fp.write(tmp)

    #--------------------------------------------------------------------------

    if not args.no_super_pixel:
        if verbose:
            print('*' * 12 + '\nComputing superpixel heatmap')

        params = Params(
            learning_rate = 0.1,
            max_iterations = 50,
            tv_beta = 3,
            l1_coeff = 0.2,
            tv_coeff = 1.,
            less_coeff = 0.,
            lasso_coeff = 4.,
            noise_sigma = 0.,
            noise_scale = 1.,
            blur = False,
            target_shape = target_shape,
            target_id = args.target_id)

        mask_path = os.path.join(out_folder, 'smooth/mask.png')
        mask_init = load_mask(mask_path)

        results = explain.compute_heatmap_using_superpixels(model, original_img, params, mask_init, HAS_CUDA, gpu_id, verbose=False)
        upsampled_mask, blurred_img_numpy, target_prob, output_prob, blurred_prob, history, class_id = results

        if verbose:
            print('Prediction drops to {:.6f}'.format(output_prob))
        out = os.path.join(out_folder, 'superpixel')
        utils.save(original_img, blurred_img_numpy, upsampled_mask, dest_folder=out, save_orig=False)

        spx_drop, spx_p = compute_stats(target_prob, output_prob, blurred_prob)
        data += ',' + str(output_prob) + ',' + str(spx_drop) + ',' + str(blurred_prob) + ',' + str(spx_p)

        # save loss history
        tmp = 'total_loss, class_loss, l1_loss, tv_loss, lasso_loss, less_loss\n'
        for t in history:
            t = [str(v) + ',' for v in t]
            t = ''.join(t) + '\n'
            tmp += t
        with open(os.path.join(out_folder, 'superpixel/loss_history.csv'), 'w') as fp:
            fp.write(tmp)
    else:
        data += ',Inf,Inf,Inf,Inf'

    #--------------------------------------------------------------------------

    if verbose:
        print('*' * 12 + '\nSaving results')

    data += '\n'
    with open(out_file, 'a') as fp:
        fp.write(data)

    t_end = time.time()
    if verbose:
        print('Elapsed time: {:.1f} seconds'.format(t_end - t_start))


if __name__ == '__main__':

    args = get_params()

    HAS_CUDA = True
    gpu_ids = [0, 1]

    '''
    HAS_CUDA = torch.cuda.is_available()
    gpu_ids = [x for x in range(torch.cuda.device_count())]
    '''

    Params = recordclass('Params', [
                                'learning_rate',
                                'max_iterations',
                                'tv_beta',
                                'l1_coeff',
                                'tv_coeff',
                                'less_coeff',
                                'lasso_coeff',
                                'noise_sigma',
                                'noise_scale',
                                'blur',
                                'target_shape',
                                'target_id'])


    if os.path.isdir(args.input_path):

        dirinfo = os.listdir(args.input_path)
        ext = args.file_ext
        filenames = [f for f in dirinfo if f.endswith(ext)]
        filenames = np.sort(filenames).tolist()
        if args.max_images is not None:
            filenames = filenames[:args.max_images]

        chunk_size = int(len(filenames) // len(gpu_ids))
        fnames = [filenames[i:i+chunk_size] for i in range(0, len(filenames), chunk_size)]


        def evaluate_on_gpu(gid, filenames):
            print('Loading model on gpu', gid)
            model_info = utils.load_model(args.modelname, HAS_CUDA, gid)

            for fname in filenames:
                try:
                    out_folder = os.path.join(args.dest_folder, os.path.basename(fname)[:-len(ext)])
                    run_evaluation(model_info, os.path.join(args.input_path, fname), out_folder, gid, verbose=args.verbose)
                except Exception as e:
                    print(e)
                    #raise e


        try:
            from joblib import Parallel, delayed
            n_threads = len(gpu_ids)
            _ = Parallel(n_jobs=n_threads)(delayed(evaluate_on_gpu)(gid, fnames[gid]) for gid in gpu_ids)
        except Exception as e:
            print(e)
            try:
                print('Try with single thread...')
                n_threads = 1
                _ = Parallel(n_jobs=n_threads)(delayed(evaluate_on_gpu)(gid, fnames[gid]) for gid in gpu_ids)
            except Exception as e:
                print(e)
                quit()


        def read_csv(filename):
            assert os.path.exists(filename)
            return np.genfromtxt(filename, delimiter=',', dtype=None)

        header = ''
        t = [None] * len(gpu_ids)
        for gid in gpu_ids:
            data = read_csv(args.results_file + '.part' + str(gid))
            header = data[0, :]
            t[gid] = data[1:, :]

        results_data = t[0]
        for i in range(1, len(gpu_ids)):
            results_data = np.vstack((results_data, t[i]))
        results_data = np.squeeze(results_data.astype(str))

        with open(args.results_file, 'w') as fp:
            t = [str(v.decode('utf-8')) + ',' for v in header]
            t = ''.join(t) + '\n'
            fp.write(t)
            for i in range(results_data.shape[0]):
                t = [str(j) +',' for j in results_data[i]]
                t = ''.join(t) + '\n'
                fp.write(t)

        for gid in gpu_ids:
            os.remove(args.results_file + '.part' + str(gid))

    else:
        print('Loading model')
        model_info = utils.load_model(args.modelname, HAS_CUDA, gpu_ids[0])

        try:
            run_evaluation(model_info, args.input_path, verbose=args.verbose)    
            os.rename(args.results_file + '.part' + str(gpu_ids[0]), args.results_file)
        except Exception as e:
            print(e)
            #raise e
