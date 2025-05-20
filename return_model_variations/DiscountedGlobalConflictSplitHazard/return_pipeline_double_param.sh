#!/bin/bash

## pass several things
## name of the corresponding job

hazardparam="$1"
rateparam="$2"
iter="$3"
jobname="$4"
dir="$5"
functype="$6"

# iteration_no=$((iteration_no + 1))
# echo $iteration_no

# bash return_pipeline.sh $hazard_rate $iteration_no

if squeue -u $USER | grep -q $jobname; then
    ## jobs with previous parameter $p$ are still running, nothing to do
    ## submit this job again
    echo "job still running for ${hazardparam} and ${rateparam} at iteration ${iter}"
    sleep 5
    #rm ret_1_param_*.out
    sbatch return_pipeline_double_param.sbatch $hazardparam $rateparam $iter $jobname $dir $functype
else
    echo "job finished for $jobname with ${hazardparam} ${rateparam}"
    if [[ $iter -gt 300 ]]; then
        echo "calibration reached max iteration"
    else
        rm ret_2_param_*.out ${jobname}*.out
        module load miniforge
        conda activate migration_env
        echo "bayesian optimization running...."
        python calibration_two_param.py --input_param1 $hazardparam  --input_param2 $rateparam --it $iter --dir_prefix $dir --functiontype $functype
        newhazardparam=$(tail -n 2 next_param_suggestion_${dir}_${functype}_${iter}.opt | head -n 1)
        newrateparam=$(tail -n 1 next_param_suggestion_${dir}_${functype}_${iter}.opt)
        echo "new evaluation value found at ${newhazardparam} ${newrateparam} ... creating new set of jobs..."
        if [[ $newhazardparam =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
            echo "valid hazard parameter."
        else
            echo "Error: A valid next hazard parameter was not found."
            exit 1
        fi
        if [[ $newrateparam =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
            echo "valid rate parameter."
        else
            echo "Error: A valid next rate parameter was not found."
            exit 1
        fi
        iter=$((iter + 1))
        python abm_dest_return_gen.py $iter $newhazardparam $functype $jobname $dir $newrateparam > pipeline_return_${functype}_${dir}_iteration_${iter}
        python create_return_dir_bayesian.py $iter $functype $dir
        conda deactivate
        bash pipeline_return_${functype}_${dir}_iteration_${iter}
        echo "new set of jobs submitted with $newhazardparam"
        sbatch return_pipeline_double_param.sbatch $newhazardparam $newrateparam $iter $jobname $dir $functype
    fi
fi
