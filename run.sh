#!/bin/bash

modelname=alexnet
outdir=results/alexnet_5k
outfile=$outdir/results.csv

mkdir -p $outdir
#echo "filename, predicted_class_id, target_prob, smooth_mask_prob, smooth_drop, smooth_blurred_prob, smooth_p, sharp_mask_prob, sharp_drop, sharp_blurred_prob, sharp_p, spx_mask_prob, spx_drop, spx_blurred_prob, spx_p" > $outfile

echo "Save results in $outdir"

indir=/media/Data/datasets/misc/imagenet_val_5k
ext=.JPEG

#indir=examples/original/
#ext=.jpg

echo "####################################"

python3 main.py --modelname $modelname --input_path $indir --dest_folder $outdir --results_file $outfile --file_ext $ext $1

echo "####################################"
echo "DONE"
