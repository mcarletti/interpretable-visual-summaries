#!/bin/bash

# --target_id <CLASS_ID> --max_images <MAX>

modelname=alexnet
outdir=/media/Data/datasets/sharp-heatmapts-pt/$modelname/$2
#outdir=results/$modelname/$2
#outdir=results/alexnet_examples_focus
#outdir=results/googlenet_5k
outfile=$outdir/results.csv

mkdir -p $outdir

#echo "Save results in $outdir"

indir=/media/Data/datasets/ImageNet/ILSVRC2012_img_train/$2
#indir=examples/focus
#indir=/media/Data/datasets/misc/imagenet_val_5k
ext=.JPEG

#echo "####################################"

python3 main.py --modelname $modelname --input_path $indir --dest_folder $outdir --results_file $outfile --file_ext $ext "$@"

#echo "####################################"
echo "DONE $2"
