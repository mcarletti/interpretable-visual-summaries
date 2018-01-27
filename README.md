# Understanding Deep Architectures by Interpretable Visual Summaries [1/2]

### [M. Carletti](http://marcocarletti.altervista.org/), M. Godi, M. Aghaei, [M. Cristani](http://profs.sci.univr.it/~cristanm/)


Project @ [author's page](http://marcocarletti.altervista.org/publications/understanding-visual-summaries/)

Paper @ [https://arxiv.org/](https://arxiv.org/)

![visual summaries](http://marcocarletti.altervista.org/publications/understanding-visual-summaries/fig1.jpg)

---
**NOTE**

The project consists of two parts. Given a set of images belonging to the same class/category, the former part generates a crisp saliency mask for each image in the set. The second part computes a set of visual summaries starting from the crisp masks.

This is the FIRST part of the project.

You can find [HERE](https://github.com/godimarcovr/interpretable_visual_summaries) the second part of the project concerning the computation of the visual summaries.

---

## Requirements
- Install [PyTorch](http://pytorch.org/) and _torchvision_ for Python 3.5
- Install the following Python 3.5 modules: _numpy_, _cv2_
- [Optional] To run `rank_regions.py` and `show_regions.py`: _matplotlib_, _skimage_
- [Optional] Download [ImageNet](http://image-net.org/download)

## Usage [1/2]: generate crisp masks

![crisp mask example](http://marcocarletti.altervista.org/content/icpr18.png)

### Example 1 (single sample):

`python3 main.py --modelname alexnet --input_path examples/original/robin3.jpg --dest_folder results/robin3 --results_file results/robin3/results.csv`

### Example 2 (generic folder):

`python3 main.py --modelname alexnet --input_path examples/original --dest_folder results/alexnet_example_original --results_file results/alexnet_example_original/results.csv --file_ext .jpg`

### Example 3 (images from the same ImageNet class):
For example, consider class _robin_ (class id = 15)

`python3 main.py --modelname alexnet --input_path <path_to>/ImageNet/ILSVRC2012_img_train/15 --dest_folder results/alexnet_imagenet --results_file results/alexnet_imagenet/results.csv --file_ext .JPEG --target_id 15 --max_images 50`

## Usage [2/2]: generate visual summaries

Follow instructions [HERE](https://github.com/godimarcovr/interpretable_visual_summaries).
