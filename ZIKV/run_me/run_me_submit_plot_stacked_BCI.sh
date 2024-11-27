#!/bin/bash

FILE=$(cd ../../ && pwd)/'main_dir.txt'
DIR=$(<$FILE)

# Edit the argument below for different input and output
export SCRIPT='/scripts/submit_plot_coverage.py'
export RUNNING_SCRIPT='/scripts/plot_stacked_CI.py'
export PARAM='logka_P logkd_P logKd_P ka_P Kd_P kd_P'
# export PARAM='logka logkd logKd ka kd Kd'

# Define the list of names
name_list="ASAP-0020915_1 ASAP-0020915_2 ASAP-0020915_3 ASAP-0020915_4"

# Loop over each name in name_list and perform commands
for NAME in $name_list; do

    export MCMC_DIR="/ZIKV/${NAME}"
    export OUT_DIR="/ZIKV/${NAME}/Stacked_BIC"

    export INCLUDE_GLOBAL_FITTING=true  # Change to false to exclude

    # Check if the output directory exists; if not, create it
    OUTPUT_DIR="$DIR$OUT_DIR"

    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"  # Create the directory, including parent directories if needed
        echo "Created directory: $OUTPUT_DIR"
    else
        echo "Directory already exists: $OUTPUT_DIR"
    fi

    # Build the command
    COMMAND="python $DIR$SCRIPT --running_script $DIR$RUNNING_SCRIPT --mcmc_dir $DIR$MCMC_DIR --out_dir $OUTPUT_DIR --parameter \"$PARAM\" "

    # Conditionally add --global_fitting if INCLUDE_GLOBAL_FITTING is true
    if [ "$INCLUDE_GLOBAL_FITTING" = true ]; then
        COMMAND="$COMMAND --global_fitting"
    fi

    # Execute the command
    eval $COMMAND

done
