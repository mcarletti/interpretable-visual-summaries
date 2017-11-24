@set outdir=results\\alexnet_original
@set outfile=%outdir%\\results.csv
@if not exist %outdir% mkdir %outdir%
@echo filename, target_prob, smooth_mask_prob, smooth_drop, smooth_blurred_prob, smooth_p, sharp_mask_prob, sharp_drop, sharp_blurred_prob, sharp_p, spx_mask_prob, spx_drop, spx_blurred_prob, spx_p > %outfile%

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
    python main.py --input_image examples/%1 --dest_folder %outdir%/%fname% --results_file %outfile% --super_pixel True
)
@goto ContinueLoop

:End
@echo ####################################
@echo DONE
