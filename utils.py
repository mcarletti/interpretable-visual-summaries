import torch
from torchvision import models
import numpy as np
import cv2
import os


def load_model(use_cuda=False):
    '''Load pretrained model.
    '''
    model = models.vgg16_bn(pretrained=True)
    model.eval()

    if use_cuda:
        model.cuda()
    
    # freeze training
    for p in model.features.parameters(): # conv layers
        p.requires_grad = False
    for p in model.classifier.parameters(): # fc layers
        p.requires_grad = False

    return model


def get_class_info(pred, label_file='ilsvrc_2012_labels.txt'):
    class_id = np.argmax(pred.cpu().data.numpy())
    class_prob = pred[0, class_id].data.cpu().squeeze().numpy()[0]
    class_labels = np.loadtxt(open(label_file), dtype=object, delimiter='\n')
    return class_id, class_prob, class_labels[class_id]


def numpy_to_torch(image, requires_grad = True, use_cuda=False):
    '''Convert numpy image to torch variable (differentiable tensor).
    '''

    # if RGB/BGR, force the float32 bit format
    if len(image.shape) < 3:
        output = np.float32([image])
    else:
        # covnert to channel-first format
        output = np.transpose(image, (2, 0, 1))

    # create a torch tensor
    output = torch.from_numpy(output)
    output.unsqueeze_(0) # make it a batch

    if use_cuda:
        output = output.cuda()

    return torch.autograd.Variable(output, requires_grad = requires_grad)


def preprocess_image(image, use_cuda=False):
    '''Prepare a cv2 image to feed a VGG classifier.
    '''

    # VGG initialization magic numbers
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    # revert the channels from BGR to RGB
    preprocessed_img = image.copy()[: , :, ::-1]

    # actual preprocessing
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    preprocessed_img_tensor = torch.from_numpy(preprocessed_img)
    preprocessed_img_tensor.unsqueeze_(0) # make it a batch

    if use_cuda:
        preprocessed_img_tensor = preprocessed_img_tensor.cuda()

    return torch.autograd.Variable(preprocessed_img_tensor, requires_grad = False)


def save(original_image, blurred_image, mask, dest_folder='results'):
    '''Save the computed images in dest_folder.
    '''
    # normalize the RGB-8bit image
    img = np.float32(original_image) / 255

    # convert to channel-last format
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))
    # normalize mask (NOTE: not between 0 and 1)
    mask = (mask - np.min(mask)) / np.max(mask)
    # revert pixel values (now, 0 = not masked; 1 = masked)
    mask = 1 - mask

    # save mask as heatmap (RGB image, JET colormap)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    
    # normalize the heatmap only for visualization purposes
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)

    # perturbate the original image by masking the pixels
    # accordingly to the computed mask
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred_image)   

    # if necessary, create the destination folder
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # save images
    cv2.imwrite(os.path.join(dest_folder, 'perturbated.png'), np.uint8(255 * perturbated))
    cv2.imwrite(os.path.join(dest_folder, 'heatmap.png'), np.uint8(255 * heatmap))
    cv2.imwrite(os.path.join(dest_folder, 'mask.png'), np.uint8(255 * mask))
    cv2.imwrite(os.path.join(dest_folder, 'cam.png'), np.uint8(255 * cam))
