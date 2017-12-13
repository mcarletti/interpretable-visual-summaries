# Sharp Heatmaps in PyTorch

Usage:

`python3 main.py --modelname MODEL --input_path IN_PATH --dest_folder OUT_PATH --results_file CSVFILE --file_ext IMG_EXT [--target_id ILSVRC2012_CLASS_ID [--max_images N]]`

This is a PyTorch implementation and extension of ***"Interpretable Explanations of Black Boxes by Meaningful Perturbation. Ruth Fong, Andrea Vedaldi"***  with some deviations.

This learns a sharp mask of pixels that helps to explain the result of a black box, e.g. a neural network.
The mask is learned by posing an optimization problem and solving directly for the mask values.

This project can use any differentiable model.
