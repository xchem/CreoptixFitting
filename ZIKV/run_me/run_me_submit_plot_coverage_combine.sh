#!/bin/bash

FILE=$(cd ../../ && pwd)/'main_dir.txt'
DIR=$(<$FILE)

export RUNNING_SCRIPT='/scripts/plot_containing_rate_of_cis_combined.py'
export PARAM='logka logkd logKd ka kd Kd'

# Define the list of names
name_list="ASAP-0020915 ASAP-0020915_2nd ASAP-0029143 ASAP-0029214"

# Loop over each name in name_list and create a SLURM job file for each
for NAME in $name_list; do

    export MCMC_DIR="/ZIKV/${NAME}"
    export NLS_FILE="/input/ZIKV/NLS_${NAME}.csv"
    export OUT_DIR="/ZIKV/NLS/${NAME}"
    OUTPUT_DIR="$DIR$OUT_DIR"
    
    # Create the output directory if it doesn't exist
    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
        echo "Created directory: $OUTPUT_DIR"
    else
        echo "Directory already exists: $OUTPUT_DIR"
    fi
    
    # Define log file paths
    log_file="${OUTPUT_DIR}/rate_of_cis.log"

    # Build the Python command
    COMMAND="python $DIR$RUNNING_SCRIPT --mcmc_dir $DIR$MCMC_DIR --out_dir $OUTPUT_DIR --nonlinear_fit_result_file $DIR$NLS_FILE --parameter \"$PARAM\""

    # Create a SLURM job file
    JOB_FILE="${OUTPUT_DIR}/rate_of_cis.job"
    
    # Write the SLURM job file
    cat <<EOT > $JOB_FILE
#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --ntasks=4
#SBATCH --mem=4096M
#SBATCH -o ${log_file}
#SBATCH -e ${log_file}

module load miniconda/3
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate gci

# Execute the command
$COMMAND
EOT
    # Submit the SLURM job
    sbatch $JOB_FILE
    echo "Submitted job for $NAME with job file $JOB_FILE"

done