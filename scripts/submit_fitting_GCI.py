import os
import sys
import argparse

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--running_script", type=str, default="")
parser.add_argument("--out_dir", type=str, default="")

parser.add_argument("--global_fitting", action="store_true", default=False)
parser.add_argument("--fitting_complex", action="store_true", default=False)
parser.add_argument("--fitting_subtract", action="store_true", default=False)

parser.add_argument("--include_protein_decay", action="store_true", default=False)
parser.add_argument("--return_conc", action="store_true", default=False)
parser.add_argument("--return_y_offset", action="store_true", default=False)
parser.add_argument("--return_epsilon", action="store_true", default=False)

parser.add_argument("--init_niters", type=int, default=1000)
parser.add_argument("--init_nburn", type=int, default=200)
parser.add_argument("--niters", type=int, default=5000)
parser.add_argument("--nburn", type=int, default=2000)
parser.add_argument("--nchain", type=int, default=4)
parser.add_argument("--random_key", type=int, default=0)

parser.add_argument("--analyte_file", type=str, default="")
parser.add_argument("--analyte_keys_included", type=str, default="")
parser.add_argument("--analyte_keys_excluded", type=str, default="")
parser.add_argument("--analyte_concentration_uM", type=str, default="10")
parser.add_argument("--calibration_file", type=str, default="")
parser.add_argument("--calibration_keys_included", type=str, default="")
parser.add_argument("--end_dissociation", type=float, default=10.0)

args = parser.parse_args()

assert os.path.exists(args.out_dir), print(args.out_dir + " does not exist.")
if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)

# Load the CSV file and extract the header row
data = pd.read_csv(args.analyte_file, sep=",")
header = data.columns.tolist()

# Filter the header to include only those that end with "Y"
header = [_cycle for _cycle in header if _cycle.endswith("Y")]

analyte_keys_excluded = args.analyte_keys_excluded.split()
key_excluded = [int(k) for k in analyte_keys_excluded]  # Convert keys to integers

# Remove all columns in header that start with '1'
filtered_header = [col for col in header if not col.startswith("Fc=1")]

# Then proceed with your existing code using filtered_header
try:
    if args.fitting_subtract:
        _cycles = [(col.split("-")[2]) for col in filtered_header]
    else:
        _cycles = [(col.split("-")[1]) for col in filtered_header]
        key_excluded.append(1)
except:
    print(
        "Please check the whether inputs from f{args.analyte_file} is subtracted sensorgram!"
    )
    sys.exit()

_cycles = [cycle.replace("_Y", "") for cycle in _cycles]
_cycles = np.array([int(cycle) for cycle in _cycles])
_cycles = _cycles[~np.isin(_cycles, key_excluded)]  # Exclude keys
cycles = np.unique(_cycles)

print("There are cycles:", cycles)

analyte_keys_included = args.analyte_keys_included.split()
calibration_keys_included = args.calibration_keys_included.split()

# Convert args.analyte_concentration_uM into list of floats or a float
try:
    analyte_values = [float(x) for x in args.analyte_concentration_uM.split()]
    if len(analyte_values) == 1:
        analyte_values = analyte_values[0]
except ValueError:
    # If conversion fails, it's a single float value
    analyte_values = float(args.analyte_concentration_uM)

# Check if analyte_values is a list or a single float
if isinstance(analyte_values, list):
    analyte_concentration_uM = np.array(analyte_values)
elif isinstance(analyte_values, float):
    analyte_concentration_uM = np.full(len(cycles), analyte_values)

N_concs = len(analyte_concentration_uM)
N_cycles = len(cycles)
assert N_concs == N_cycles, print(
    "Incorrect input. There is f{N_concs} analyte concentrations while f{N_cycles} cycles."
)

if args.global_fitting:
    global_fitting = " --global_fitting "
else:
    global_fitting = ""

if args.fitting_complex:
    fitting_complex = " --fitting_complex "
else:
    fitting_complex = ""

if args.fitting_subtract:
    fitting_subtract = " --fitting_subtract "
else:
    fitting_subtract = ""

if args.include_protein_decay:
    include_protein_decay = " --include_protein_decay "
else:
    include_protein_decay = ""

if args.return_conc:
    return_conc = " --return_conc "
else:
    return_conc = ""

if args.return_y_offset:
    return_y_offset = " --return_y_offset "
else:
    return_y_offset = ""

if args.return_epsilon:
    return_epsilon = " --return_epsilon "
else:
    return_epsilon = ""

if len(cycles) > 0:
    for idx, j in enumerate(cycles):
        if args.global_fitting:
            out_dir = os.path.join(args.out_dir, f"{j}_Y")
        else:
            out_dir = args.out_dir

        _key_included = []
        _calibration_keys_included = []

        if args.fitting_subtract:
            for ith_analyte, ith_calib in zip(
                analyte_keys_included, calibration_keys_included
            ):
                _key_included.append([f"Fc={ith_analyte}-1-{j}_Y"])
                _calibration_keys_included.append([f"Fc={ith_calib}_Y"])
        else:
            for ith_analyte, ith_calib in zip(
                analyte_keys_included, calibration_keys_included
            ):
                _key_included.append([f"Fc={ith_analyte}-{j}_Y"])
                _calibration_keys_included.append([f"Fc={ith_calib}_Y"])

        if args.fitting_complex or (not args.fitting_subtract):
            _key_included_FC1 = f"Fc=1-{j}_Y"
        else:
            _key_included_FC1 = ""

        # Check if the items in _key_included are lists and flatten if needed
        # Flatten any nested lists if present
        _key_included = [
            item if isinstance(item, str) else item[0] for item in _key_included
        ]
        _calibration_keys_included = [
            item if isinstance(item, str) else item[0]
            for item in _calibration_keys_included
        ]

        analyte_keys = " ".join(_key_included)
        calibration_keys = " ".join(_calibration_keys_included)

        qsub_file = os.path.join(args.out_dir, f"{j}.job")
        log_file = os.path.join(args.out_dir, f"{j}.log")

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
cd """
            + args.out_dir
            + """\n"""
            + """date\n"""
            + """python """
            + args.running_script
            + """ --out_dir """
            + out_dir
            + global_fitting
            + fitting_complex
            + fitting_subtract
            + include_protein_decay
            + return_conc
            + return_y_offset
            + return_epsilon
            + """ --init_niters %d """ % args.init_niters
            + """ --init_nburn %d """ % args.init_nburn
            + """ --niters %d """ % args.niters
            + """ --nburn %d """ % args.nburn
            + """ --nchain %d """ % args.nchain
            + """ --random_key %d """ % args.random_key
            + """ --analyte_file """
            + args.analyte_file
            + ''' --analyte_keys_included "%s"''' % analyte_keys
            + """ --analyte_keys_included_FC1 "%s" """ % _key_included_FC1
            + """ --analyte_concentration_uM %0.5f""" % analyte_concentration_uM[idx]
            + """ --calibration_file """
            + args.calibration_file
            + ''' --calibration_keys_included "%s"''' % calibration_keys
            + """ --end_dissociation %0.1f""" % args.end_dissociation
            + """\ndate\n"""
        )

        print("Writing qsub file", qsub_file)
        open(qsub_file, "w").write(qsub_script)
        os.system("sbatch %s" % qsub_file)
