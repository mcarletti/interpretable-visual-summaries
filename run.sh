#!/bin/bash

modelname=googlenet
outdir=results/googlenet_5k
outfile=$outdir/results.csv
mkdir -p $outdir
echo "filename, target_prob, smooth_mask_prob, smooth_drop, smooth_blurred_prob, smooth_p, sharp_mask_prob, sharp_drop, sharp_blurred_prob, sharp_p, spx_mask_prob, spx_drop, spx_blurred_prob, spx_p" > $outfile

echo "Save results in $outdir"

for image in "/media/Data/datasets/misc/imagenet_val_5k"/*.JPEG
#for image in "examples/original"/*.jpg
do
    echo "####################################"
    fname=`echo $image | cut -d '/' -f 7 | cut -d '.' -f 1`
    #fname=`echo $image | cut -d '/' -f 3 | cut -d '.' -f 1`
    if [[ "$fname" == *"masked"* ]]; then
        echo "Skipping $fname"
    else
        echo "Processing $image"
        python3 main.py --modelname $modelname --input_image $image --dest_folder $outdir/$fname --results_file $outfile --super_pixel
    fi
done

echo "####################################"
echo "DONE"
