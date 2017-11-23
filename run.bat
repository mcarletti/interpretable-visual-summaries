@set outdir=results\\vgg16_bn
@set outfile=%outdir%\\results.csv
@if not exist %outdir% mkdir %outdir%
@echo filename, target_prob, smooth_mask_prob, smooth_drop, sharp_mask_prob, sharp_drop, sharp/smooth drop ratio > %outfile%

@echo Save results in %outdir%

@for %%f in (examples/*.jpg) do (
    @call :Iter %%f
    :ContinueLoop
    @rem continue
)
@goto End

:Iter
@echo ####################################
@set fname=%1
@set fname=%fname:~0,-4%
@if not "%fname%"=="%fname:masked=%" (
    echo Skipping %fname%
) else (
    echo Processing %fname%
    python explain.py --input_image examples/%1 --dest_folder %outdir%/%fname% --results_file %outfile%
)
@goto ContinueLoop

:End
@echo ####################################
@echo DONE
