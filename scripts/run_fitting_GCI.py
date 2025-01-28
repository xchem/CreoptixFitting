import sys
import os
import argparse

import jax
import jax.numpy as jnp
import numpyro

from _loading_data import _load_gci_infor_from_argument, _extract_GCI
from _fitting_GCI import fitting_expts

import mrich

parser = argparse.ArgumentParser()

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
parser.add_argument("--analyte_keys_included_FC1", type=str, default="")
parser.add_argument("--analyte_concentration_uM", type=float, default="10")
parser.add_argument("--calibration_file", type=str, default="")
parser.add_argument("--calibration_keys_included", type=str, default="")
parser.add_argument("--end_dissociation", type=float, default=10.0)

parser.add_argument("--logka_C", type=float, default=10.78)
parser.add_argument("--logKd_C", type=float, default=-10.25)
parser.add_argument("--Rmax_C", type=float, default=187.98)
parser.add_argument("--alpha", type=float, default=jnp.nan)

args = parser.parse_args()

print(args)

jax.config.update("jax_enable_x64", True)
numpyro.set_host_device_count(4)

GAS_CONSTANT = 1.9872041  # the gas constant, in cal/mol/K

analyte_keys_included = args.analyte_keys_included.split(";")
calibration_keys_included = args.calibration_keys_included.split(";")

print(analyte_keys_included)
print(calibration_keys_included)

assert len(analyte_keys_included) == len(calibration_keys_included), print(
    "Please check both provided keys."
)

if not args.fitting_subtract:
    assert len(args.analyte_keys_included_FC1) > 0, print("Please provide FC1 key.")

if args.global_fitting:
    init_expts = []
    for analyte_keys, calibration_keys in zip(
        analyte_keys_included, calibration_keys_included
    ):
        init_expts.append(
            _load_gci_infor_from_argument(analyte_keys, calibration_keys, args)
        )
    experiments = _extract_GCI(init_expts, args)

    _, params_hat = fitting_expts(experiments, args.out_dir, args)

else:
    for analyte_keys, calibration_keys in zip(
        analyte_keys_included, calibration_keys_included
    ):
        init_expts = []
        init_expts.append(
            _load_gci_infor_from_argument(analyte_keys, calibration_keys, args)
        )
        experiments = _extract_GCI(init_expts, args)
        _, params_hat = fitting_expts(
            experiments, os.path.join(args.out_dir, analyte_keys[3:]), args
        )
        del init_expts
