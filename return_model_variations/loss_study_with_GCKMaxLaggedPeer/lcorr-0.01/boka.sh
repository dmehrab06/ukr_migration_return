#!/bin/bash

## pass several things
## name of the corresponding job

jobname="$1"

if squeue -u $USER | grep -q $jobname; then
    echo "job still running"
else
    echo "job finished"
fi
