import argparse
import torch
import numpy as np
import cv2
import os

import utils
from regularizers import l1_reg, tv_reg, lasso_reg

from collections import namedtuple


Params = namedtuple('Params', [
                            'learning_rate',
                            'max_iterations',
                            'tv_beta',
                            'l1_coeff',
                            'tv_coeff',
                            'lasso_coeff',
                            'noise_scale',
                            'target_shape'])


def compute_heatmap(model, original_img, params, mask_init, use_cuda=False):
    '''Compute image heatmaps according to: https://arxiv.org/abs/1704.03296
    Interpretable Explanations of Black Boxes by Meaningful Perturbation

    Params:
        model           : deep neural network or other black box model; e.g. VGG
        params          : namedtuple of settings
        original_img    : input image, RGB-8bit
        mask_init       : init heatmap
        use_cuda        : enable/disable GPU usage
    '''

    # scale between 0 and 1 with 32-bit color depth
    img = np.float32(original_img) / 255

    # generate a perturbated version of the input image as the mean
    # between a gaussian-blurred and median-blurred version of the
    # original image
    blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
    blurred_img2 = np.float32(cv2.medianBlur(original_img, 11)) / 255
    # this image is used to mask the original image
    blurred_img_numpy = (blurred_img1 + blurred_img2) / 2
    
    # prepare image to feed to the model
    img = utils.preprocess_image(img, use_cuda) # original image
    blurred_img = utils.preprocess_image(blurred_img2, use_cuda) # blurred version of input image
    mask = utils.numpy_to_torch(mask_init, use_cuda=use_cuda) # init mask

    upsample = torch.nn.Upsample(size=params.target_shape, mode='bilinear')

    if use_cuda:
        upsample.cuda()

    # optimize only the heatmap
    optimizer = torch.optim.Adam([mask], lr=params.learning_rate)

    # compute the target output
    targets = torch.nn.Softmax()(model(img))
    category, target_prob, label = utils.get_class_info(targets)
    print("Category with highest probability:", (label, category, target_prob))

    print("Optimizing.. ")
    for i in range(params.max_iterations):

        # upsample the mask and use it
        # the mask is duplicated to have 3 channels since it is
        # single channel and is used with a 224*224 RGB image
        # NOTE: the upsampled mask is only used to compute the
        # perturbation on the input image
        upsampled_mask = upsample(mask)
        upsampled_mask = upsampled_mask.expand(1, 3, *params.target_shape)
        
        # use the (upsampled) mask to perturbated the input image
        # blend the median blurred image and the original (scaled) image
        # accordingly to the current (upsampled) mask
        perturbated_input = img.mul(upsampled_mask) + \
                            blurred_img.mul(1 - upsampled_mask)
        
        # gaussian noise with is added to the preprocssed image
        # at each iteration, inspired by google's smooth gradient
        # https://arxiv.org/abs/1706.03825
        # https://pair-code.github.io/saliency/
        noise = np.zeros(params.target_shape + (3,), dtype = np.float32)
        noise = noise + cv2.randn(noise, 0., 0.2)
        noise = utils.numpy_to_torch(noise, use_cuda=use_cuda)
        noisy_perturbated_input = perturbated_input + noise * params.noise_scale
        
        # compute current prediction
        outputs = torch.nn.Softmax()(model(noisy_perturbated_input))
        output_prob = outputs[0, category]

        # compute the loss and use the regularizers
        loss = output_prob + \
                params.l1_coeff * l1_reg(mask) + \
                params.tv_coeff * tv_reg(mask, params.tv_beta) + \
                params.lasso_coeff * lasso_reg(mask)

        # update the optimization process
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # optional: clamping seems to give better results
        # should be useless, but numerical s**t happens
        mask.data.clamp_(0, 1)

    # upsample the computed final mask
    upsampled_mask = upsample(mask)

    # compute the prediction probabilities before
    # and after the perturbation and masking
    outputs = torch.nn.Softmax()(model(perturbated_input))
    output_prob = outputs[0, category].data.cpu().squeeze().numpy()[0]

    return upsampled_mask, blurred_img_numpy, target_prob, output_prob


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True)
    parser.add_argument('--dest_folder', type=str, required=True)
    parser.add_argument('--results_file', type=str, required=True)
    args = parser.parse_args()

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
        max_iterations = 500,
        tv_beta = 3,      # exponential tv factor
        l1_coeff = 0.01,  # reduces number of masked pixels
        tv_coeff = 0.2,   # encourages compact and smooth heatmaps
        lasso_coeff = 0., # force masked pixels to binary values
        noise_scale = 1., # scale factor of additional perturbation
        target_shape = target_shape)

    # assume to mask all the pixels (all ones)
    # as described in the paper, the resolution of the initial mask
    # is lower than the processed image because we want to avoid
    # artifacts in the optimization process; this causes also a "natural"
    # mask which is blurred due to the upsampling
    mask_init = np.ones((28, 28), dtype = np.float32)

    results = compute_heatmap(model, original_img, params, mask_init, use_cuda)
    upsampled_mask, blurred_img_numpy, target_prob, output_prob = results

    print('Prediction drops to {:.6f}'.format(output_prob))
    out = os.path.join(args.dest_folder, 'smooth')
    utils.save(original_img, blurred_img_numpy, upsampled_mask, dest_folder=out)

    #data = 'filename, target_prob, smooth_mask_prob, smooth_drop, sharp_mask_prob, sharp_drop, sharp/smooth drop ratio\n'
    smooth_drop = (target_prob - output_prob) / target_prob
    data = args.input_image + ',' + str(target_prob) + ',' + str(output_prob) + ',' + str(smooth_drop)

    print('*' * 12)
    print('Computing sharp heatmap')

    params = Params(
        learning_rate = 0.1,
        max_iterations = 500,
        tv_beta = 7,
        l1_coeff = 0.075,
        tv_coeff = 2.,
        lasso_coeff = 1.,
        noise_scale = 10.,
        target_shape = target_shape)

    # we want to generate a sharp heatmap to simplify the region
    # explanation and object detection/segmentation
    # since working on a high resolution mask causes scattered
    # results, we use as initialization mask the low resolution
    # output from the previous optimization (reference paper)
    mask_path = os.path.join(args.dest_folder, 'smooth/mask.png')
    mask_init = cv2.imread(mask_path, 1)
    mask_init = cv2.cvtColor(mask_init, cv2.COLOR_BGR2GRAY)
    mask_init = np.float32(mask_init) / 255
    mask_init = 1. - mask_init # revert the activations

    results = compute_heatmap(model, original_img, params, mask_init, use_cuda)
    upsampled_mask, blurred_img_numpy, target_prob, output_prob = results

    print('Prediction drops to {:.6f}'.format(output_prob))
    out = os.path.join(args.dest_folder, 'sharp')
    utils.save(original_img, blurred_img_numpy, upsampled_mask, dest_folder=out)

    sharp_drop = (target_prob - output_prob) / target_prob
    data += ',' + str(output_prob) + ',' + str(sharp_drop) + ',' + str(sharp_drop/smooth_drop) + '\n'
    with open(args.results_file, 'a') as fp:
        fp.write(data)
