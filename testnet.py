import argparse
from PIL import Image
import numpy as np
import os

import torch
from torch.autograd import Variable
from torchvision.models import alexnet, vgg16, inception_v3, resnet50
import torchvision.transforms as transforms

MODELS = {'alexnet': alexnet,   'googlenet': inception_v3, 'vgg': vgg16,     'resnet': resnet50}
SHAPES = {'alexnet': (224,224), 'googlenet': (299,299),    'vgg': (224,224), 'resnet': (224,224)}

def numpy_to_torch(array, use_cuda=False, gpu_id=0, enable_grads=True, as_batch=False, to_float32=True):
    L = len(array.shape)
    # convert vectors to float32 arrays
    # when the array has one dimension, we do not
    # add channel information
    if L == 1:
        array = np.float32(array) if to_float32 else array
    # add channels information to 2D arrays
    # any torch tensor will have the first dimension
    # corresponding to the number of channels
    elif L == 2:
        array = np.array([array])
    # we assume that the internal format of the array
    # is HWC, so we need to move the channels in the
    # first position
    elif L == 3:
        array = np.transpose(array, (2,0,1))
    else:
        raise ValueError('Input array must be 1, 2 or 3d')
    # convert to float32 if asked
    array = np.float32(array) if to_float32 else array
    # avoid non negative strides: data is stored orderly
    # BUG: PyTorch does not yet support negative strides
    array = np.ascontiguousarray(array)
    tensor = torch.from_numpy(array)
    # we expand and/or move to gpu the tensor data
    # BEFORE converting it to a Variable so it can be
    # used as leaf-tensor in the optimization graph
    if as_batch:
        tensor.unsqueeze_(0)
    if use_cuda:
        tensor = tensor.cuda(gpu_id)
    # create variable with/without gradients
    tensor = Variable(tensor, requires_grad=enable_grads)
    return tensor

def torch_to_numpy(tensor):
    # make the tensor a variable to be able to
    # access its data without throwing exceptions
    if not isinstance(tensor, Variable):
        tensor = Variable(tensor)
    # move to cpu and convert the data to a numpy array 
    array = tensor.cpu().data.numpy()
    # remove useless dimensions
    array = array.squeeze()
    # return a scalar if the array has one single value
    array = np.asscalar(array) if array.size == 1 else array
    return array

def softmax(arr):
    e = np.exp(arr)
    s = e / np.sum(e)
    return s, np.argmax(s)

def preprocess_image(ndarray, target_shape, transf=None):
    if len(ndarray.shape) == 2:
        ndarray = np.stack([ndarray] * 3, axis=-1)
    image = Image.fromarray(ndarray)

    if transf is None:
        mu = (0.485, 0.456, 0.406)
        sd = (0.229, 0.224, 0.225)
        transf = transforms.Compose([transforms.Resize(target_shape),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mu, sd)])

    image = transf(image)
    return image

def test_image(model, image, target_shape, use_cuda=False, gpu_id=0):
    '''Test one single image, which could be a PyTorch FloatTensor (CxHxW) or a numpy.ndarray (HxW or HxWxC).
    '''
    if isinstance(image, np.ndarray):
        image = preprocess_image(image, target_shape)

    if use_cuda:
        image = image.cuda(gpu_id)

    model.eval()
    pred = model(Variable(image.unsqueeze(0)))
    pred = torch_to_numpy(pred).squeeze()
    pred, class_id = softmax(pred)

    return pred, class_id

if __name__ == '__main__':

    seed = 23092017
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #%% Check arguments ------------------------------------------------------------

    # Parse arguments.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='alexnet')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    assert args.model in MODELS.keys()

    if args.gpu_id is not None:
        args.use_cuda = True
    
    if args.use_cuda:
        if args.gpu_id is None:
            args.gpu_id = 0

    if args.verbose:
        print(args)

    try:
        model = MODELS[args.model](pretrained=True)
        model.eval()

        if args.use_cuda:
            model.cuda(args.gpu_id)
        
        if args.verbose:
            print(model)
        
        target_shape = SHAPES[args.model]

        filename = 'path_to_image'
        image = np.asarray(Image.open(filename).resize(target_shape))

        predictions, class_id = test_image(model=model, image=image, target_shape=target_shape, use_cuda=args.use_cuda, gpu_id=args.gpu_id)
        print('Prediction: class', class_id, ' (', predictions[class_id], ')')

    except Exception as e:
        #print(e.with_traceback, e.args)
        raise e
