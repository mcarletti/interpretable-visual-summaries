import argparse
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
                            'target_shape'])


def load_mask(filename):
    mask_init = cv2.imread(filename, 1)
    mask_init = cv2.cvtColor(mask_init, cv2.COLOR_BGR2GRAY)
    mask_init = np.float32(mask_init) / 255
    mask_init = 1. - mask_init # revert the activations
    return mask_init


if __name__ == '__main__':

    t_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='examples/original/robin3.jpg')
    parser.add_argument('--dest_folder', type=str, default='results/robin3')
    parser.add_argument('--results_file', type=str, default='results/robin3/results.csv')
    parser.add_argument('--super_pixel', action='store_true')

    args = parser.parse_args()

    data = ''
    if not os.path.exists(args.results_file):
        data = 'filename, target_prob, smooth_mask_prob, smooth_drop, smooth_blurred_prob, smooth_p, sharp_mask_prob, sharp_drop, sharp_blurred_prob, sharp_p, spx_mask_prob, spx_drop, spx_blurred_prob, spx_p\n'

    use_cuda = torch.cuda.is_available()

    print('Loading model')
    model = utils.load_model(use_cuda)
    target_shape = (224, 224)

    # load the BGR image, resize it to the right resolution
    original_img = cv2.imread(args.input_image, 1)
    original_img = cv2.resize(original_img, target_shape)

    print('*' * 12)
    print('Computing blurred heatmap')

    params = Params(
        learning_rate = 0.1,
        max_iterations = 300,
        tv_beta = 3,      # exponential tv factor
        l1_coeff = 0.01,  # reduces number of masked pixels
        tv_coeff = 0.2,   # encourages compact and smooth heatmaps
        less_coeff = 0.,  # encourages similarity between predictions
        lasso_coeff = 0., # force masked pixels to binary values
        noise_sigma = 0., # sigma of additional perturbation
        noise_scale = 1., # scale factor of additional perturbation
        target_shape = target_shape)

    # assume to mask all the pixels (all ones)
    # as described in the paper, the resolution of the initial mask
    # is lower than the processed image because we want to avoid
    # artifacts in the optimization process; this causes also a "natural"
    # mask which is blurred due to the upsampling
    mask_init = np.ones((28, 28), dtype = np.float32)

    results = explain.compute_heatmap(model, original_img, params, mask_init, use_cuda)
    upsampled_mask, blurred_img_numpy, target_prob, output_prob, blurred_prob = results

    print('Prediction drops to {:.6f}'.format(output_prob))
    out = os.path.join(args.dest_folder, 'smooth')
    utils.save(original_img, blurred_img_numpy, upsampled_mask, dest_folder=out)

    smooth_drop = (target_prob - output_prob) / target_prob
    smooth_p = (output_prob - target_prob) / (target_prob - blurred_prob)
    data += args.input_image + ',' + str(target_prob) + ',' + str(output_prob) + ',' + str(smooth_drop) + ',' + str(blurred_prob) + ',' + str(smooth_p)

    print('*' * 12)
    print('Computing sharp heatmap')

    params.tv_beta = 7
    params.l1_coeff = 0.075
    params.lasso_coeff = 1.

    # we want to generate a sharp heatmap to simplify the region
    # explanation and object detection/segmentation
    # since working on a high resolution mask causes scattered
    # results, we use as initialization mask the low resolution
    # output from the previous optimization (reference paper)
    mask_path = os.path.join(args.dest_folder, 'smooth/mask.png')
    mask_init = load_mask(mask_path)

    results = explain.compute_heatmap(model, original_img, params, mask_init, use_cuda)
    upsampled_mask, blurred_img_numpy, target_prob, output_prob, blurred_prob = results

    print('Prediction drops to {:.6f}'.format(output_prob))
    out = os.path.join(args.dest_folder, 'sharp')
    utils.save(original_img, blurred_img_numpy, upsampled_mask, dest_folder=out)

    sharp_drop = (target_prob - output_prob) / target_prob
    sharp_p = (output_prob - target_prob) / (target_prob - blurred_prob)
    data += ',' + str(output_prob) + ',' + str(sharp_drop) + ',' + str(blurred_prob) + ',' + str(sharp_p)

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

    nb_random_tests = 200
    rand_drops = np.zeros((nb_random_tests, 1), dtype=np.float32)
    for i in range(nb_random_tests):
        rand_drops[i] = compute_random_perturbation()
    print('*' * 12)
    print('Drop over', nb_random_tests, 'random perturbation [mean, var]:', (np.mean(rand_drops), np.var(rand_drops)))

    if args.super_pixel:
        print('*' * 12)
        print('Computing superpixel heatmap')

        params.tv_beta = 3
        params.l1_coeff = 0.05
        params.tv_coeff = 0.5
        params.lasso_coeff = 2.

        mask_path = os.path.join(args.dest_folder, 'sharp/mask.png')
        mask_init = load_mask(mask_path)

        results = explain.compute_heatmap_using_superpixels(model, original_img, params, mask_init, use_cuda)
        upsampled_mask, blurred_img_numpy, target_prob, output_prob, blurred_prob = results

        print('Prediction drops to {:.6f}'.format(output_prob))
        out = os.path.join(args.dest_folder, 'superpixel')
        utils.save(original_img, blurred_img_numpy, upsampled_mask, dest_folder=out)

        spx_drop = (target_prob - output_prob) / target_prob
        spx_p = (output_prob - target_prob) / (target_prob - blurred_prob)
        data += ',' + str(output_prob) + ',' + str(spx_drop) + ',' + str(blurred_prob) + ',' + str(spx_p)
    else:
        data += ',Inf,Inf,Inf,Inf'

    print('*' * 12)
    print('Saving results')

    data += '\n'
    with open(args.results_file, 'a') as fp:
        fp.write(data)

    t_end = time.time()
    print('Elapsed time: {:.1f} seconds'.format(t_end - t_start))
