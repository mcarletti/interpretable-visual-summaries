import torch
import utils
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label as labelize
from skimage.measure import regionprops


def get_params():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str, default='alexnet')
    parser.add_argument('--impath', type=str, default='/media/Data/datasets/sharp-heatmapts-pt/235')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--cuda', type=bool, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


def get_crop_and_chull(im, prop):
    bbox = prop.bbox
    convhull_bw = prop.convex_image # same size of bbox
    cropped_region = im[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    if cropped_region.shape[2] == 3:
        convhull_bw = np.asarray([convhull_bw] * 3)
        convhull_bw = np.transpose(convhull_bw, (1, 2, 0))
    return cropped_region, convhull_bw


args = get_params()

HAS_CUDA = args.cuda
if HAS_CUDA is None:
    HAS_CUDA = torch.cuda.is_available()

print('Loading model:', args.modelname)
model, target_shape = utils.load_model(args.modelname, use_cuda=HAS_CUDA)


dirinfo = os.listdir(args.impath)
dirs = [os.path.join(args.impath, d) for d in dirinfo if os.path.isdir(os.path.join(args.impath, d))]

dirs = dirs[:50]

print("Found", len(dirs), "folders")
nb_img_to_compute = len(dirs)


def predict(net, x, class_id=None):
    outputs = torch.nn.Softmax()(model(x))
    if class_id is not None:
        outputs = outputs[0, class_id] # probs
        outputs = outputs.data.cpu().squeeze().numpy()[0]
    return outputs


global_prop_saliency = []
global_doa = []


for i, d in enumerate(dirs[:nb_img_to_compute]):

    print('Processing', d)

    if args.verbose:
        print('Loading and preprocessing input image and mask')
    # load input image
    original_img = cv2.imread(os.path.join(d, 'smooth/original.png'), 1)
    original_img = cv2.resize(original_img, target_shape)
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    scaled_img = np.float32(original_img) / 255

    # generate a perturbated version of the input image
    blurred_img_numpy = cv2.GaussianBlur(scaled_img, (11, 11), 10)

    # load mask
    mask_numpy_ref = cv2.imread(os.path.join(d, 'smooth/mask.png'), 1)
    mask_numpy_ref = 1. - np.float32(mask_numpy_ref[:, :, 0]) / 255
    mask_bw_ref = np.uint8(255. - mask_numpy_ref * 255.)
    #_, mask_bw_ref = cv2.threshold(mask_bw_ref, 132, 255, cv2.THRESH_BINARY)
    _, mask_bw_ref = cv2.threshold(mask_bw_ref, 204, 255, cv2.THRESH_BINARY)

    mask_numpy = cv2.imread(os.path.join(d, 'sharp/mask.png'), 1)
    mask_numpy = 1. - np.float32(mask_numpy) / 255
    # binarize mask
    # revert the image to correctly compute the regions
    mask_bw = np.uint8(255. - mask_numpy[:, :, 0] * 255.)
    _, mask_bw = cv2.threshold(mask_bw, 128, 255, cv2.THRESH_BINARY)

    # convert images to torch tensors
    img = utils.preprocess_image(scaled_img, use_cuda=HAS_CUDA)
    blurred_img = utils.preprocess_image(blurred_img_numpy, use_cuda=HAS_CUDA)
    mask = utils.numpy_to_torch(mask_numpy, use_cuda=HAS_CUDA)

    if args.verbose:
        print('Computing classification confidence drop')
    target_probs = predict(model, img, None)
    category, target_prob, label = utils.get_class_info(target_probs)
    if args.verbose:
        print('Category with highest probability:', (label, category, target_prob))

    perturbated_input = img.mul(mask) + blurred_img.mul(1 - mask)
    perturbated_prob = predict(model, perturbated_input, category)
    if args.verbose:
        print('Confidence after perturbing the input image: {:.6f}'.format(perturbated_prob))

    '''
    1. extract regions from mask
    2. for each region, perturbate the original image and compute the drop
    3. sort the regions according to the drop
    4. show the first k regions
    5. compute the drop-over-area score per region
    '''

    if args.verbose:
        print('Computing region proposals')


    def eval_regions(bw, thr=0.0):
        mask_labeled = labelize(bw)
        regions = regionprops(mask_labeled)

        prop_saliency = []
        doa = []
        min_area = int(np.prod(target_shape) * thr)
        valid_idx = []
        for pid, props in enumerate(regions):
            if props.area < min_area:
                continue
            valid_idx.append(pid)
            # extract proposal mask
            prop_mask = np.ones(bw.shape, dtype=np.float32)
            for u, v in props.coords:
                prop_mask[u, v] = 0.
            # compute contribution
            prop_mask = utils.numpy_to_torch(prop_mask, use_cuda=HAS_CUDA)
            perturbated_input = img.mul(prop_mask) + blurred_img.mul(1 - prop_mask)
            drop = 1. - predict(model, perturbated_input, category)
            prop_saliency.append(drop)
            doa.append(drop / props.area)
            #print('Region saliency: {:.6f}'.format(prop_saliency[-1]))

        prop_saliency = np.asarray(prop_saliency)
        doa = np.asarray(doa)
        regions = np.asarray(regions)

        idx = np.argsort(prop_saliency)[::-1][:args.top_k]
        prop_saliency = prop_saliency[idx]
        doa[idx]
        regions = regions[valid_idx][idx]

        return regions, prop_saliency, doa


    
    regions, prop_saliency, doa = eval_regions(mask_bw, 0.01)

    # save per the specific image the set of bbox
    '''
    regions_ref, _, _ = eval_regions(mask_bw_ref, 0.01)
    if len(regions_ref) > 0:

        print('Save bbox:', d)

        with open(os.path.join(d, 'smooth/bbox.csv'), 'w') as fp:
            for prop in regions_ref:
                bb = prop.bbox
                bb = [str(b) for b in bb]
                fp.write(bb[0] + ',' + bb[1] + ',' + bb[2] + ',' + bb[3] + '\n')

        with open(os.path.join(d, 'sharp/bbox.csv'), 'w') as fp:
            for prop in regions:
                bb = prop.bbox
                bb = [str(b) for b in bb]
                fp.write(bb[0] + ',' + bb[1] + ',' + bb[2] + ',' + bb[3] + '\n')

        quit()'''

    panned_prop_saliency = np.zeros((args.top_k,), dtype=np.float32)
    panned_doa = np.zeros((args.top_k,), dtype=np.float32)
    for j in range(prop_saliency.shape[0]):
        panned_prop_saliency[j] = prop_saliency[j]
        panned_doa[j] = doa[j]
    global_prop_saliency.append(panned_prop_saliency)
    global_doa.append(panned_doa)

    plt.figure(0)

    rows = args.top_k + 2 # for original image and reference mask (Vedaldi)
    cols = min(12, nb_img_to_compute)
    nb_regions = min(args.top_k, regions.shape[0])

    parent = '/media/Data/datasets/sharp-heatmapts-pt/cropped_regions/' + str(category)
    for k in range(nb_regions):

        crop, chull = get_crop_and_chull(rgb_img, regions[k])

        # save the top-1 region in the specified folder
        '''top_path = os.path.join(parent, 'top_' + str(k))
        if not os.path.exists(top_path):
            os.makedirs(top_path)

        # save cropped region from rgb image
        rgb_path = os.path.join(top_path, 'rgb/')
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)
        fname = os.path.basename(d) + '.png'
        cv2.imwrite(os.path.join(rgb_path, fname), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        # save bbox coords
        bbox = regions[k].bbox
        with open(os.path.join(top_path, 'bbox.csv'), 'a') as fp:
            bb = [',' + str(b) for b in bbox]
            fp.write(fname + ''.join(bb) + '\n')

        # save cropped region from binary mask
        mask_path = os.path.join(top_path, 'mask/')
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)
        cropped_mask = mask_bw[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        cv2.imwrite(os.path.join(mask_path, fname), cropped_mask)

        mask_path = os.path.join(top_path, 'mask_big/')
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)
        prop_mask = np.zeros(mask_bw.shape, dtype=np.uint8)
        for u, v in regions[k].coords:
            prop_mask[u, v] = 255
        cv2.imwrite(os.path.join(mask_path, fname), prop_mask)'''

        # visualize only the specified number of columns
        if i >= cols:
            continue

        plt.subplot(rows, cols, i + 1)
        plt.imshow(rgb_img)
        #plt.imshow(1. - mask_bw_ref, alpha=0.5, cmap=plt.get_cmap('binary'))
        plt.imshow(1. - mask_numpy_ref, alpha=0.5, cmap=plt.get_cmap('jet'))

        plt.subplot(rows, cols, i + 1 + cols)
        plt.imshow(rgb_img)
        plt.imshow(1. - mask_bw, alpha=0.5, cmap=plt.get_cmap('binary'))

        plt.subplot(rows, cols, i + 1 + cols + cols * (k + 1))
        plt.imshow(crop * chull)
        #plt.imshow(crop_ref * chull_ref)
        '''err = 1. - (mask_bw + mask_bw_ref) / 2.
        err = np.uint8(255. - err * 255.)
        _, err = cv2.threshold(err, 128, 255, cv2.THRESH_BINARY)
        plt.imshow(err)'''
        plt.text(0, 0, 'Drop: {:.3f}'.format(prop_saliency[k]), bbox=dict(facecolor='red', alpha=0.5))
        #plt.text(0, 0, 'DoA:  {:.3f}'.format(doa[k]), bbox=dict(facecolor='blue', alpha=0.5))

global_prop_saliency = np.asarray(global_prop_saliency)
global_doa = np.asarray(global_doa)


def show_graphs(title, data, show=False):
    plt.figure()
    plt.subplot(1, 2, 1)
    mu = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    xx = np.linspace(1, args.top_k, args.top_k)
    plt.plot(xx, mu, 'r', label='mean')
    plt.plot(xx, mu - sd, 'b--', label='mu +/- sd')
    plt.plot(xx, mu + sd, 'b--')
    plt.legend()
    plt.title(title)

    plt.subplot(1, 2, 2)
    plt.boxplot(data)
    #plt.title(title)


show_graphs('K-top saliency (drop)', global_prop_saliency)
show_graphs('K-top Drop-over-Area (DoA)', global_doa)

plt.tight_layout()
plt.show()
