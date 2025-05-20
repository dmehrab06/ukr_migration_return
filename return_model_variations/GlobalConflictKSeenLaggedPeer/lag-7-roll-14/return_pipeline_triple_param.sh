#!/bin/bash

## pass several things
## name of the corresponding job

hazardparam="$1"
rateparam="$2"
peerthresh="$3"
iter="$4"
jobname="$5"
dir="$6"
functype="$7"

# iteration_no=$((iteration_no + 1))
# echo $iteration_no

# bash return_pipeline.sh $hazard_rate $iteration_no

if squeue -u $USER | grep -q $jobname; then
    ## jobs with previous parameter $p$ are still running, nothing to do
    ## submit this job again
    echo "job still running for ${hazardparam} ${rateparam} ${peerthresh} at iteration ${iter}"
    sleep 5
    #rm ret_1_param_*.out
    sbatch return_pipeline_triple_param.sbatch $hazardparam $rateparam $peerthresh $iter $jobname $dir $functype
else
    echo "job finished for $jobname with ${hazardparam} ${rateparam} ${peerthresh}"
    if [[ $iter -gt 200 ]]; then
        #rm ret_3_param_*.out ${jobname}*.out
        echo "calibration reached max iteration"
    else
        #rm ${jobname}*.out
        module load miniforge
        conda activate migration_env
        echo "bayesian optimization running...."
        python calibration_three_param.py --input_param1 $hazardparam  --input_param2 $rateparam --input_param3 $peerthresh --it $iter --dir_prefix $dir --functiontype $functype
        newhazardparam=$(tail -n 3 next_param_suggestion_${dir}_${functype}_${iter}.opt | head -n 1)
        newrateparam=$(tail -n 2 next_param_suggestion_${dir}_${functype}_${iter}.opt | head -n 1)
        newpeerthresh=$(tail -n 1 next_param_suggestion_${dir}_${functype}_${iter}.opt)
        echo "new evaluation value found at ${newhazardparam} ${newrateparam} ${newpeerthresh} ... creating new set of jobs..."
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
        if [[ $newpeerthresh =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
            echo "valid peer threshold."
        else
            echo "Error: A valid next rate parameter was not found."
            exit 1
        fi
        iter=$((iter + 1))
        python abm_dest_return_gen.py $iter $newhazardparam $functype $jobname $dir $newrateparam $newpeerthresh > pipeline_return_${functype}_${dir}_iteration_${iter}
        python create_return_dir_bayesian.py $iter $functype $dir
        conda deactivate
        bash pipeline_return_${functype}_${dir}_iteration_${iter}
        echo "new set of jobs submitted with ${newhazardparam} ${newrateparam} ${newpeerthresh}"
        cp *.pkl  ~/Migration_Projects/radiation_model/migration_shock/scripts/theory_of_planned_behavior_migration/return/optimizers/
        sbatch return_pipeline_triple_param.sbatch $newhazardparam $newrateparam $newpeerthresh $iter $jobname $dir $functype
    fi
fi
