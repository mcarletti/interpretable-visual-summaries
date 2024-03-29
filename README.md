# Understanding Deep Architectures by Visual Summaries [1/2]

### M. Godi, [M. Carletti](http://marcocarletti.altervista.org/), M. Aghaei, F. Giuliari, [M. Cristani](http://profs.sci.univr.it/~cristanm/)

Paper @ [BMVC](http://bmvc2018.org/contents/papers/0794.pdf) or [ARXIV](https://arxiv.org/abs/1801.09103)

---
**NOTE**

The project consists of two parts. Given a set of images belonging to the same class/category, the former part generates a crisp saliency mask for each image in the set. The second part computes a set of visual summaries starting from the crisp masks.

This is the FIRST part of the project.

You can find [HERE](https://github.com/godimarcovr/interpretable_visual_summaries) the second part of the project concerning the computation of the visual summaries.

---

## Requirements
To generate crisp saliency maps (first part) you need to install the following libraries:
- [PyTorch](http://pytorch.org/) and _torchvision_ for Python 3.5
- Python 3.5 modules: _numpy_, _cv2_
- [Optional] To run `rank_regions.py` and `show_regions.py`: _matplotlib_, _skimage_
- [Optional] Download [ImageNet](http://image-net.org/download)

To generate a set of visual summaries (second part) for a specified class you need to follow instructions [HERE](https://github.com/godimarcovr/interpretable_visual_summaries).

## Usage [1/2]: generate crisp masks

![crisp mask example](http://marcocarletti.altervista.org/content/bmvc18_visual.png)

### Example 1 (single sample):

`python3 main.py --modelname alexnet --input_path examples/original/robin3.jpg --dest_folder results/robin3 --results_file results/robin3/results.csv`

### Example 2 (generic folder):

`python3 main.py --modelname alexnet --input_path examples/original --dest_folder results/alexnet_example_original --results_file results/alexnet_example_original/results.csv --file_ext .jpg`

### Example 3 (images from the same ImageNet class):
For example, consider class _robin_ (class id = 15)

`python3 main.py --modelname alexnet --input_path <path_to>/ImageNet/ILSVRC2012_img_train/15 --dest_folder results/alexnet_imagenet --results_file results/alexnet_imagenet/results.csv --file_ext .JPEG --target_id 15 --max_images 50`

## Usage [2/2]: generate visual summaries

Follow instructions [HERE](https://github.com/godimarcovr/interpretable_visual_summaries).
