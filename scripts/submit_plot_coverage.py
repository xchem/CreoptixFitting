import os
import sys
import argparse

# Use module name for logger
import logging

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--running_script", type=str, default="")
parser.add_argument("--mcmc_dir", type=str, default="")
parser.add_argument("--out_dir", type=str, default="")

parser.add_argument("--parameter", type=str, default="logka logkd logKd")
parser.add_argument("--central", type=str, default="median")

parser.add_argument("--global_fitting", action="store_true", default=False)

args = parser.parse_args()

parameters = args.parameter.split()
print("Params:", parameters)

if args.global_fitting:
    global_fitting = " --global_fitting "
else:
    global_fitting = ""

for param in parameters:
    qsub_file = os.path.join(args.out_dir, f"{param}.job")
    log_file = os.path.join(args.out_dir, f"{param}.log")

    qsub_script = (
        """#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --ntasks=4
#SBATCH --mem=4096M
#SBATCH -o %s """
        % log_file
        + """
#SBATCH -e %s """
        % log_file
        + """

module load miniconda/3
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate gci \n
date\n
python """
        + args.running_script
        + """ --mcmc_dir """
        + args.mcmc_dir
        + """ --out_dir """
        + args.out_dir
        + """ --parameter "%s" """ % param
        + global_fitting
        + """ --central """
        + args.central
        + """\ndate\n"""
    )

    print("Writing qsub file", qsub_file)
    open(qsub_file, "w").write(qsub_script)
    print("Submitting " + param)
    os.system("sbatch %s" % qsub_file)
