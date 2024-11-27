#!/bin/bash

FILE=$(cd ../../ && pwd)/'main_dir.txt'
DIR=$(<$FILE)

# Edit the argument below for different input and output

export SCRIPT='/scripts/submit_fitting_GCI.py'
export RUNNING_SCRIPT='/scripts/run_fitting_GCI.py'

export NAME='ASAP-0020915_2'
export OUT_DIR='/ZIKV'

export INCLUDE_GLOBAL_FITTING=true  # Change to false to exclude
export FITTING_COMPLEX=false        # Change to false to exclude
export FITTING_SUBTRACT=true       # Change to false to exclude
export INCLUDE_PROTEIN_DECAY=false  # Change to false to exclude
export INCLUDE_Y_OFFSET=true        # Change to false to exclude
export INCLUDE_NOISE=false          # Change to false to exclude

export ANALYTE_FILE="/input/ZIKV/ZIKV_${NAME}_Subtraction.csv"
export ANALYTE_KEYS='2 3 4'
export EXCLUDE_KEYS=''
export CONC='10'
export DMSO_FILE="/input/ZIKV/ZIKV_DMSO_2nd.csv"
export DMSO_KEYS='2-11 3-11 4-11'
export END_T=10

export INIT_NITERS=1000
export INIT_NBURNS=200
export NITERS=5000
export NBURNS=2000

# Check if the output directory exists; if not, create it
OUTPUT_DIR="$DIR$OUT_DIR/$NAME"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"  # Create the directory, including parent directories if needed
    echo "Created directory: $OUTPUT_DIR"
else
    echo "Directory already exists: $OUTPUT_DIR"
fi

# Build the command
COMMAND="python $DIR$SCRIPT --running_script $DIR$RUNNING_SCRIPT --out_dir $OUTPUT_DIR --analyte_file $DIR$ANALYTE_FILE --analyte_keys_included \"$ANALYTE_KEYS\" --analyte_keys_excluded \"$EXCLUDE_KEYS\" --analyte_concentration_uM \"$CONC\" --calibration_file $DIR$DMSO_FILE --calibration_keys_included \"$DMSO_KEYS\" --init_niters $INIT_NITERS --init_nburn $INIT_NBURNS --niters $NITERS --nburn $NBURNS --end_dissociation $END_T"

# Conditionally add --global_fitting if INCLUDE_GLOBAL_FITTING is true
if [ "$INCLUDE_GLOBAL_FITTING" = true ]; then
    COMMAND="$COMMAND --global_fitting"
fi

# Conditionally add --fitting_complex if FITTING_COMPLEX is true
if [ "$FITTING_COMPLEX" = true ]; then
    COMMAND="$COMMAND --fitting_complex"
fi

# Conditionally add --fitting_subtract if FITTING_SUBTRACT is true
if [ "$FITTING_SUBTRACT" = true ]; then
    COMMAND="$COMMAND --fitting_subtract"
fi

# Conditionally add --include_protein_decay if INCLUDE_PROTEIN_DECAY is true
if [ "$INCLUDE_PROTEIN_DECAY" = true ]; then
    COMMAND="$COMMAND --include_protein_decay"
fi

# Conditionally add --y_offset if INCLUDE_Y_OFFSET is true
if [ "$INCLUDE_Y_OFFSET" = true ]; then
    COMMAND="$COMMAND --return_y_offset"
fi

# Conditionally add --return_epsilon if INCLUDE_NOISE is true
if [ "$INCLUDE_NOISE" = true ]; then
    COMMAND="$COMMAND --return_epsilon"
fi

# Execute the command
eval $COMMAND
