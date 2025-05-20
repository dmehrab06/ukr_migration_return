#!/bin/bash

## pass several things
## name of the corresponding job

hazardparam="$1"
iter="$2"
jobname="$3"
dir="$4"
functype="$5"

# iteration_no=$((iteration_no + 1))
# echo $iteration_no

# bash return_pipeline.sh $hazard_rate $iteration_no

if squeue -u $USER | grep -q $jobname; then
    ## jobs with previous parameter $p$ are still running, nothing to do
    ## submit this job again
    echo "job still running for $hazardparam at iteration $iter"
    sleep 5
    #rm ret_1_param_*.out
    sbatch return_pipeline_single_param.sbatch $hazardparam $iter $jobname $dir $functype
else
    echo "job finished for $jobname with $hazardparam"
    if [[ $iter -gt 50 ]]; then
        echo "calibration reached max iteration"
	rm ret_1_param*.out ${jobname}*.out
    else
        rm ret_1_param_*.out ${jobname}*.out
        module load miniforge
        conda activate migration_env
        echo "bayesian optimization running...."
        python calibration.py --input_param $hazardparam --it $iter --dir_prefix $dir --functiontype $functype
        newhazardparam=$(tail -n 1 next_param_suggestion_${dir}_${functype}_${iter}.opt)
        echo "new evaluation value found at $newhazardparam ... creating new set of jobs..."
        if [[ $newhazardparam =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
            echo "valid parameter."
        else
            echo "Error: A valid next parameter was not found."
            exit 1
        fi
        iter=$((iter + 1))
        python abm_dest_return_gen.py $iter $newhazardparam $functype $jobname $dir > pipeline_return_${functype}_${dir}_iteration_${iter}
        python create_return_dir_bayesian.py $iter $functype $dir
        conda deactivate
        bash pipeline_return_${functype}_${dir}_iteration_${iter}
        echo "new set of jobs submitted with $newhazardparam"
        cp *.pkl  ~/Migration_Projects/radiation_model/migration_shock/scripts/theory_of_planned_behavior_migration/return/optimizers
        sbatch return_pipeline_single_param.sbatch $newhazardparam $iter $jobname $dir $functype
    fi
fi
