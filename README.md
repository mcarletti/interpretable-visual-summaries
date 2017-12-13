# Sharp Heatmaps in PyTorch

This is a PyTorch implementation and extension of ***"Interpretable Explanations of Black Boxes by Meaningful Perturbation. Ruth Fong, Andrea Vedaldi"***  with some deviations.

This learns a sharp mask of pixels that helps to explain the result of a black box, e.g. a neural network.
The mask is learned by posing an optimization problem and solving directly for the mask values.

This project can use any differentiable model.

## Usage

`python3 main.py --modelname MODEL --input_path IN_PATH --dest_folder OUT_PATH --results_file CSVFILE --file_ext IMG_EXT [--target_id ILSVRC2012_CLASS_ID [--max_images N]]`

### Example 1 (genereic folder):

`python3 main.py --modelname alexnet --input_path examples/original --dest_folder results/alexnet_example_original --results_file results/alexnet_example_original/results.csv --file_ext .jpg`

### Example 2 (images from the same class):

`python3 main.py --modelname alexnet --input_path <path_to>/ImageNet/ILSVRC2012_img_train/234 --dest_folder results/alexnet_imagenet --results_file results/alexnet_imagenet/results.csv --file_ext .JPEG --target_id 234 --max_images 50`

### Example 3 (single sample):

`python3 main.py --modelname alexnet --input_path examples/original/robin3.jpg --dest_folder results/robin3 --results_file results/robin3/results.csv`
