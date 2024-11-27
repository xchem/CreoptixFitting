#!/bin/bash

FILE=$(cd ../../ && pwd)/'main_dir.txt'
DIR=$(<$FILE)

# Edit the argument below for different input and output
export RUNNING_SCRIPT='/scripts/plot_coverage_combined.py'

# Define the list of names
name_list="ASAP-0020915 ASAP-0029143 ASAP-0029214"

# Loop over each name in name_list and perform commands
for NAME in $name_list; do

    export MCMC_DIR="/ZIKV/${NAME}"
    export NLS_FILE="/input/ZIKV/NLS_${NAME}.csv"
    export OUT_DIR="/ZIKV/NLS/${NAME}"

    # Check if the output directory exists; if not, create it
    OUTPUT_DIR="$DIR$OUT_DIR"

    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"  # Create the directory, including parent directories if needed
        echo "Created directory: $OUTPUT_DIR"
    else
        echo "Directory already exists: $OUTPUT_DIR"
    fi

	# Build the command
	COMMAND="python $DIR$RUNNING_SCRIPT --mcmc_dir $DIR$MCMC_DIR --out_dir $OUTPUT_DIR --nonlinear_fit_result_file $DIR$NLS_FILE"

	# Execute the command
	eval $COMMAND

done