import torch
import numpy as np
import cv2
import os
import time

import explain
import utils
from regularizers import l1_reg, tv_reg, less_reg, lasso_reg

#from collections import namedtuple
from recordclass import recordclass # mutable version of namedtuple


HAS_CUDA = torch.cuda.is_available()

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
                            'target_shape'])


def get_params():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname',    type=str, default='alexnet')
    parser.add_argument('--input_image',  type=str, default='examples/original/flute3.jpg')
    parser.add_argument('--dest_folder',  type=str, default='results/tmp')
    parser.add_argument('--results_file', type=str, default='results/tmp/results.csv')
    parser.add_argument('--super_pixel',  action='store_false')
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
            perturbed_pixels = x*x + y*y > radius*radius
            mask_init[perturbed_pixels] = 1.
    return mask_init


def compute_stats(target, output, blurred):
    drop = (target - output) / target
    p = (output - target) / (target - blurred)
    return drop, p


if __name__ == '__main__':

    #--------------------------------------------------------------------------

    t_start = time.time()

    args = get_params()

    data = ''
    if not os.path.exists(args.results_file):
        data = 'filename, target_prob, smooth_mask_prob, smooth_drop, smooth_blurred_prob, smooth_p, sharp_mask_prob, sharp_drop, sharp_blurred_prob, sharp_p, spx_mask_prob, spx_drop, spx_blurred_prob, spx_p\n'

    print('Loading model')
    model, target_shape = utils.load_model(args.modelname, HAS_CUDA)

    # load the BGR image, resize it to the right resolution
    original_img = cv2.imread(args.input_image, 1)
    original_img = cv2.resize(original_img, target_shape)

    #--------------------------------------------------------------------------

    print('*' * 12 + '\nComputing blurred heatmap')

    params = Params(
        learning_rate = 0.1,
        max_iterations = 100,
        tv_beta = 3,      # exponential tv factor
        l1_coeff = 0.01,  # 1e-4,  # reduces number of masked pixels
        tv_coeff = 0.2,   # 1e-2,  # encourages compact and smooth heatmaps
        less_coeff = 0.,  # encourages similarity between predictions
        lasso_coeff = 0., # force masked pixels to binary values
        noise_sigma = 0., # sigma of additional perturbation
        noise_scale = 1., # scale factor of additional perturbation
        blur = True,      # blur upsampled mask during optimization
        target_shape = target_shape)

    # assume to mask all the pixels (all ones)
    # as described in the paper, the resolution of the initial mask
    # is lower than the processed image because we want to avoid
    # artifacts in the optimization process; this causes also a "natural"
    # mask which is blurred due to the upsampling
    resize_factor = 8
    mask_shape = (target_shape[0] // resize_factor, target_shape[1] // resize_factor)
    mask_init = load_mask(None, mask_shape, 'circular')

    results = explain.compute_heatmap(model, original_img, params, mask_init, HAS_CUDA)
    upsampled_mask, blurred_img_numpy, target_prob, output_prob, blurred_prob, history = results

    print('Prediction drops to {:.6f}'.format(output_prob))
    out = os.path.join(args.dest_folder, 'smooth')
    utils.save(original_img, blurred_img_numpy, upsampled_mask, dest_folder=out)

    smooth_drop, smooth_p = compute_stats(target_prob, output_prob, blurred_prob)
    data += args.input_image + ',' + str(target_prob) + ',' + str(output_prob) + ',' + str(smooth_drop) + ',' + str(blurred_prob) + ',' + str(smooth_p)

    # save loss history
    tmp = 'total_loss, class_loss, l1_loss, tv_loss, lasso_loss, less_loss\n'
    for t in history:
        t = [str(v) + ',' for v in t]
        t = ''.join(t) + '\n'
        tmp += t
    with open(os.path.join(args.dest_folder, 'smooth/loss_history.csv'), 'w') as fp:
        fp.write(tmp)

    #--------------------------------------------------------------------------

    print('*' * 12 + '\nComputing sharp heatmap')

    params.max_iterations = 50
    params.learning_rate = 0.5
    params.tv_beta = 7
    params.l1_coeff = 0.1
    params.lasso_coeff = 1.
    params.blur = False

    # we want to generate a sharp heatmap to simplify the region
    # explanation and object detection/segmentation
    # since working on a high resolution mask causes scattered
    # results, we use as initialization mask the low resolution
    # output from the previous optimization (reference paper)
    mask_path = os.path.join(args.dest_folder, 'smooth/mask.png')
    mask_init = load_mask(mask_path)

    results = explain.compute_heatmap(model, original_img, params, mask_init, HAS_CUDA, verbose=False)
    upsampled_mask, blurred_img_numpy, target_prob, output_prob, blurred_prob, history = results

    print('Prediction drops to {:.6f}'.format(output_prob))
    out = os.path.join(args.dest_folder, 'sharp')
    utils.save(original_img, blurred_img_numpy, upsampled_mask, dest_folder=out)

    sharp_drop, sharp_p = compute_stats(target_prob, output_prob, blurred_prob)
    data += ',' + str(output_prob) + ',' + str(sharp_drop) + ',' + str(blurred_prob) + ',' + str(sharp_p)

    # save loss history
    tmp = 'total_loss, class_loss, l1_loss, tv_loss, lasso_loss, less_loss\n'
    for t in history:
        t = [str(v) + ',' for v in t]
        t = ''.join(t) + '\n'
        tmp += t
    with open(os.path.join(args.dest_folder, 'sharp/loss_history.csv'), 'w') as fp:
        fp.write(tmp)

    #--------------------------------------------------------------------------

    '''
    def compute_random_perturbation():
        # binarize mask, compute the number of pixels and generate a square mask
        # to randomly perturbate the pixels of the image
        mask = upsampled_mask.data.cpu().squeeze().numpy()
        mask = cv2.threshold(mask, 0.1, 1., cv2.THRESH_BINARY)[1]
        nb_perturbed_pixels = np.sum(mask)
        h_size = int(np.sqrt(nb_perturbed_pixels) * 0.5)
        x_pos = np.random.randint(h_size + 1, target_shape[0] - 1)
        y_pos = np.random.randint(h_size + 1, target_shape[1] - 1)
        new_mask = np.zeros(target_shape, dtype=np.float32)
        new_mask[x_pos-h_size:x_pos+h_size, y_pos-h_size:y_pos+h_size] = 1.
        new_mask = utils.numpy_to_torch(new_mask, use_cuda=use_cuda)

        # compute the perturbation score using the random mask
        img = np.float32(original_img) / 255
        img = utils.preprocess_image(img, use_cuda)
        target_preds = model(img)
        targets = torch.nn.Softmax()(target_preds)
        category, target_prob, label = utils.get_class_info(targets)

        blurred_img = utils.preprocess_image(blurred_img_numpy, use_cuda)
        perturbated_input = img.mul(new_mask) + blurred_img.mul(1 - new_mask)
        outputs = torch.nn.Softmax()(model(perturbated_input))
        output_prob = outputs[0, category].data.cpu().squeeze().numpy()[0]

        return (target_prob - output_prob) / target_prob

    nb_random_tests = 100
    rand_drops = np.zeros((nb_random_tests, 1), dtype=np.float32)
    for i in range(nb_random_tests):
        rand_drops[i] = compute_random_perturbation()
    print('*' * 12)
    print('Drop over', nb_random_tests, 'random perturbation [mean, var]:', (np.mean(rand_drops), np.var(rand_drops)))
    '''

    #--------------------------------------------------------------------------

    if args.super_pixel:
        print('*' * 12 + '\nComputing superpixel heatmap')

        params.tv_beta = 3
        params.l1_coeff = 0.15
        params.lasso_coeff = 2.

        mask_path = os.path.join(args.dest_folder, 'smooth/mask.png')
        mask_init = load_mask(mask_path)

        results = explain.compute_heatmap_using_superpixels(model, original_img, params, mask_init, HAS_CUDA, verbose=False)
        upsampled_mask, blurred_img_numpy, target_prob, output_prob, blurred_prob, history = results

        print('Prediction drops to {:.6f}'.format(output_prob))
        out = os.path.join(args.dest_folder, 'superpixel')
        utils.save(original_img, blurred_img_numpy, upsampled_mask, dest_folder=out)

        spx_drop, spx_p = compute_stats(target_prob, output_prob, blurred_prob)
        data += ',' + str(output_prob) + ',' + str(spx_drop) + ',' + str(blurred_prob) + ',' + str(spx_p)

        # save loss history
        tmp = 'total_loss, class_loss, l1_loss, tv_loss, lasso_loss, less_loss\n'
        for t in history:
            t = [str(v) + ',' for v in t]
            t = ''.join(t) + '\n'
            tmp += t
        with open(os.path.join(args.dest_folder, 'superpixel/loss_history.csv'), 'w') as fp:
            fp.write(tmp)
    else:
        data += ',Inf,Inf,Inf,Inf'

    #--------------------------------------------------------------------------

    print('*' * 12 + '\nSaving results')

    data += '\n'
    with open(args.results_file, 'a') as fp:
        fp.write(data)

    t_end = time.time()
    print('Elapsed time: {:.1f} seconds'.format(t_end - t_start))
