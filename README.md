# Sharp Heatmaps in PyTorch

Usage:

`python explain.py --input_image <path_to_image> --dest_folder <path_to_folder> --results_file <path_to_csv_file>`

This is a PyTorch implementation and extension of ***"Interpretable Explanations of Black Boxes by Meaningful Perturbation. Ruth Fong, Andrea Vedaldi"***  with some deviations.

This learns a mask of pixels that explain the result of a black box.
The mask is learned by posing an optimization problem and solving directly for the mask values.

This project can use any differentiable model.
