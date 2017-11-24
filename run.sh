#!/bin/bash

outdir=results/alexnet
outfile=$outdir/results.csv
mkdir -p $outdir
echo "filename, target_prob, smooth_mask_prob, smooth_drop, smooth_blurred_prob, smooth_p, sharp_mask_prob, sharp_drop, sharp_blurred_prob, sharp_p" > $outfile

echo "Save results in $outdir"

for image in "examples/original"/*.jpg
do
    echo "####################################"
    fname=`echo $image | cut -d '/' -f 3 | cut -d '.' -f 1`
    if [[ "$fname" == *"masked"* ]]; then
        echo "Skipping $fname"
    else
        echo "Processing $image"
        python3 explain.py --input_image $image --dest_folder $outdir/$fname --results_file $outfile
    fi
done

echo "####################################"
echo "DONE"
