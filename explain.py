import torch
from torch.autograd import Variable
import numpy as np
import cv2

import utils
from regularizers import l1_reg, tv_reg, less_reg, lasso_reg


def compute_heatmap(model, original_img, params, mask_init, use_cuda=False, gpu_id=0, verbose=False):
    '''Compute image heatmaps according to: https://arxiv.org/abs/1704.03296
    Interpretable Explanations of Black Boxes by Meaningful Perturbation

    Params:
        model           : deep neural network or other black box model; e.g. VGG
        params          : namedtuple/recordclass of settings
        original_img    : input image, RGB-8bit
        mask_init       : init heatmap
        use_cuda        : enable/disable GPU usage
    '''

    # scale between 0 and 1 with 32-bit color depth
    img = np.float32(original_img) / 255

    # generate a perturbated version of the input image
    blurred_img_numpy = cv2.GaussianBlur(img, (11, 11), 10)
    
    # prepare image to feed to the model
    img = utils.preprocess_image(img, use_cuda, gpu_id=gpu_id) # original image
    blurred_img = utils.preprocess_image(blurred_img_numpy, use_cuda, gpu_id=gpu_id) # blurred version of input image
    mask = utils.numpy_to_torch(mask_init, use_cuda=use_cuda, gpu_id=gpu_id) # init mask

    upsample = torch.nn.Upsample(size=params.target_shape, mode='bilinear')
    blur = utils.BlurTensor(use_cuda, gpu_id=gpu_id)

    if use_cuda:
        upsample = upsample.cuda(gpu_id)

    # optimize only the heatmap
    optimizer = torch.optim.Adam([mask], lr=params.learning_rate)

    # compute the target output
    target_preds = model(img)
    targets = torch.nn.Softmax()(target_preds)
    category, target_prob, label = utils.get_class_info(targets)
    if verbose:
        print("Category with highest probability:", (label, category, target_prob))

    loss_history = []

    if verbose:
        print("Optimizing.. ")
    for i in range(params.max_iterations):

        # upsample the mask and use it
        # the mask is duplicated to have 3 channels since it is
        # single channel and is used with a 224*224 RGB image
        # NOTE: the upsampled mask is only used to compute the
        # perturbation on the input image
        upsampled_mask = upsample(mask)
        if params.blur:
            upsampled_mask = blur(upsampled_mask, 5)
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
        if params.noise_sigma != 0:
            noise = noise + cv2.randn(noise, 0., params.noise_sigma)
        noise = utils.numpy_to_torch(noise, use_cuda=use_cuda, gpu_id=gpu_id)
        noisy_perturbated_input = perturbated_input + noise * params.noise_scale
        
        # compute current prediction
        preds = model(noisy_perturbated_input)
        outputs = torch.nn.Softmax()(preds)

        # compute the loss and use the regularizers
        class_loss = outputs[0, category]
        l1_loss = params.l1_coeff * l1_reg(mask)
        tv_loss = params.tv_coeff * tv_reg(mask, params.tv_beta)
        lasso_loss = params.lasso_coeff * lasso_reg(mask)
        less_loss = params.less_coeff * less_reg(preds, target_preds)

        losses = [class_loss, l1_loss, tv_loss, lasso_loss, less_loss]
        total_loss = np.sum(losses)

        # convert loss tensors to scalars
        losses = [total_loss.data.cpu().squeeze().numpy()[0]] + [l.data.cpu().numpy()[0] for l in losses]
        loss_history.append(losses)

        # update the optimization process
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # optional: clamping seems to give better results
        # should be useless, but numerical s**t happens
        mask.data.clamp_(0, 1)

    # upsample the computed final mask
    upsampled_mask = upsample(mask)
    if params.blur:
        upsampled_mask = blur(upsampled_mask, 5)

    perturbated_input = img.mul(upsampled_mask) + \
                        blurred_img.mul(1 - upsampled_mask)

    # compute the prediction probabilities before
    # and after the perturbation and masking
    outputs = torch.nn.Softmax()(model(perturbated_input))
    output_prob = outputs[0, category].data.cpu().squeeze().numpy()[0]

    # compute the prediction on the completely blurred image
    outputs = torch.nn.Softmax()(model(blurred_img))
    blurred_prob = outputs[0, category].data.cpu().squeeze().numpy()[0]

    return upsampled_mask, blurred_img_numpy, target_prob, output_prob, blurred_prob, np.asarray(loss_history), category


# Compute heatmaps using superpixels


from skimage.segmentation import slic


class Superpixel2Pixel(torch.nn.Module):

    def __init__(self, segm_img, use_cuda, gpu_id=0):
        super(Superpixel2Pixel, self).__init__()
        self.segm_img = torch.from_numpy(np.asarray(segm_img, dtype=np.int64))
        self.use_cuda = use_cuda
        self.gpu_id = gpu_id

    def forward(self, input):
        pixelmask = input[self.segm_img.cuda(self.gpu_id).view(-1)].view(self.segm_img.size(0), self.segm_img.size(1))
        return pixelmask


def compute_heatmap_using_superpixels(model, original_img, params, mask_init=None, use_cuda=False, gpu_id=0, verbose=False):

    img = np.float32(original_img) / 255
    blurred_img_numpy = cv2.GaussianBlur(img, (11, 11), 10)

    # associate at each pixel the id of the corresponding superpixel
    segm_img = slic(img.copy()[: , :, ::-1], n_segments=2000, compactness=10, sigma=0.5)
    s2p = Superpixel2Pixel(segm_img, use_cuda, gpu_id=gpu_id)

    # generate superpixel initialization image
    nb_segms = np.max(segm_img) + 1
    segm_init = np.zeros((nb_segms,), dtype=np.float32)
    if mask_init is None:
        segm_init = segm_init + 0.5
    else:
        for i in range(nb_segms):
            segm_init[i] = np.mean(mask_init[segm_img == i])
            # segm_init[i] = 0.5 if segm_init[i] < 0.5 else segm_init[i]

    blur = utils.BlurTensor(use_cuda, gpu_id=gpu_id)
    
    # create superpixel image mask
    if use_cuda:
        segm = Variable(torch.from_numpy(segm_init).cuda(gpu_id), requires_grad=True)
    else:
        segm = Variable(torch.from_numpy(segm_init), requires_grad=True)
    
    img = utils.preprocess_image(img, use_cuda, gpu_id=gpu_id) # original image
    blurred_img = utils.preprocess_image(blurred_img_numpy, use_cuda, gpu_id=gpu_id) # blurred version of input image

    optimizer = torch.optim.Adam([segm], lr=params.learning_rate)
    
    target_preds = model(img)
    targets = torch.nn.Softmax()(target_preds)
    category, target_prob, label = utils.get_class_info(targets)
    if verbose:
        print("Category with highest probability:", (label, category, target_prob))

    loss_history = []

    if verbose:
        print("Optimizing.. ")
    for i in range(params.max_iterations):
        upsampled_mask = s2p(segm).unsqueeze(0).unsqueeze(0)
        if params.blur:
            upsampled_mask = blur(upsampled_mask, 5)
        upsampled_mask = upsampled_mask.expand(1, 3, *params.target_shape)

        perturbated_input = img.mul(upsampled_mask) + \
                            blurred_img.mul(1 - upsampled_mask)

        noise = np.zeros(params.target_shape + (3,), dtype = np.float32)
        if params.noise_sigma != 0:
            noise = noise + cv2.randn(noise, 0., params.noise_sigma)
        noise = utils.numpy_to_torch(noise, use_cuda=use_cuda, gpu_id=gpu_id)
        noisy_perturbated_input = perturbated_input + noise * params.noise_scale
        
        preds = model(noisy_perturbated_input)
        outputs = torch.nn.Softmax()(preds)

        current_mask = segm # upsampled_mask

        class_loss = outputs[0, category]
        l1_loss = params.l1_coeff * l1_reg(current_mask)
        tv_loss = params.tv_coeff * tv_reg(upsampled_mask, params.tv_beta)
        lasso_loss = params.lasso_coeff * lasso_reg(current_mask)
        less_loss = params.less_coeff * less_reg(preds, target_preds)

        losses = [class_loss, l1_loss, tv_loss, lasso_loss, less_loss]
        total_loss = np.sum(losses)

        losses = [total_loss.data.cpu().squeeze().numpy()[0]] + [l.data.cpu().numpy()[0] for l in losses]
        loss_history.append(losses)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        segm.data.clamp_(0, 1)

    if params.blur:
        upsampled_mask = blur(upsampled_mask, 5)

    perturbated_input = img.mul(upsampled_mask) + \
                        blurred_img.mul(1 - upsampled_mask)
                        
    outputs = torch.nn.Softmax()(model(perturbated_input))
    output_prob = outputs[0, category].data.cpu().squeeze().numpy()[0]

    outputs = torch.nn.Softmax()(model(blurred_img))
    blurred_prob = outputs[0, category].data.cpu().squeeze().numpy()[0]

    return upsampled_mask, blurred_img_numpy, target_prob, output_prob, blurred_prob, np.asarray(loss_history), category
