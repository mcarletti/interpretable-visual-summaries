#!/bin/bash

outdir=results/vgg16_bn
outfile=$outdir/results.csv
mkdir -p $outdir
echo "filename, target_prob, smooth_mask_prob, smooth_drop, sharp_mask_prob, sharp_drop, sharp/smooth drop ratio" > $outfile

echo "Save results in $outdir"

for image in "examples"/*.jpg
do
    echo "####################################"
    fname=`echo $image | cut -d '/' -f 2 | cut -d '.' -f 1`
    if [[ "$fname" == *"masked"* ]]; then
        echo "Skipping $fname"
    else
        echo "Processing $image"
        python3 explain.py --input_image $image --dest_folder $outdir/$fname --results_file $outfile
    fi
done

echo "####################################"
echo "DONE"
