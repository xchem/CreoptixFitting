import os
import glob
import argparse
import pickle

import numpy as np
import jax.numpy as jnp

from _confidence_intervals import rate_of_containing_from_sample, rate_of_containing_from_means_stds
from _plot_confidence_intervals import plot_containing_rates

parser = argparse.ArgumentParser()

parser.add_argument( "--mcmc_dir",                          type=str,               default="")
parser.add_argument( "--out_dir",                           type=str,               default="")

parser.add_argument( "--parameter",                         type=str,               default="")
parser.add_argument( "--central",                           type=str,               default="median")

args = parser.parse_args()

assert args.central in ["mean", "median"], "wrong central"
os.chdir(args.out_dir)

LEVELS_PERCENT = np.linspace(10., 95., num=18)
print("LEVELS_PERCENT", LEVELS_PERCENT)
LEVELS = LEVELS_PERCENT/100.

mcmc_trace_files = glob.glob(os.path.join(args.mcmc_dir, '*'))
experiment_names = [os.path.basename(file) for file in mcmc_trace_files if os.path.isfile(os.path.join(file, "traces.pickle"))]

param = args.parameter
XLABEL_rename = {"logka": "$lnk_a$", "logkd": "$lnk_d$", "logKd": "$lnK_d$",
                 "logka_P": "$lnk_{a,P}$", "logkd_P": "$lnk_{d,P}$", "logKd_P": "$lnK_{d,P}$",
                 "logka_C": "$lnk_{a,C}$", "logkd_C": "$lnk_{d,C}$", "logKd_C": "$lnK_{d,C}$",
                 "ka": "$k_a$", "kd": "$k_d$", "ka_P": "$k_{a,P}$", "kd_P": "$k_{d,P}$"}

XLABEL = "Predicted " + XLABEL_rename[param]
YLABEL = "Observed "  + XLABEL_rename[param]
print("Running:" + param)

if len(mcmc_trace_files)>0:
    
    if param.startswith('logkd') or param.startswith('kd'): param_name = param+'_1'
    else: param_name = param

    if param.startswith('log'):
        if param == 'logkd':
            samples = [pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logka']+pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logKd'] for name in experiment_names]
        elif param == 'logkd_P':
            samples = [pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logka_P']+pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logKd_P'] for name in experiment_names]
        elif param == 'logkd_C':
            samples = [pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logka_C']+pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logKd_C'] for name in experiment_names]
        else:
            samples = [pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))[param] for name in experiment_names]
    else:
        if param == 'kd':
            samples = [jnp.exp(pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logka']+pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logKd']) for name in experiment_names]
        elif param == 'kd_P':
            samples = [jnp.exp(pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logka_P']+pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logKd_P']) for name in experiment_names]
        elif param == 'kd_C':
            samples = [jnp.exp(pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logka_C']+pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logKd_C']) for name in experiment_names]
        else:
            samples = [jnp.exp(pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['log'+param]) for name in experiment_names]

    if args.central == "median":
        b_centrals = [ np.median( [np.median(sample) for sample in samples] ) ]
    elif args.central == "mean":
        b_centrals = [ np.mean( [ np.mean(sample) for sample in samples] ) ]

    if not os.path.isfile(f"{param_name}.pkl"):
        b_rates = []
        b_rate_errors = []
        for level in LEVELS:
            print(level)
            rate, rate_error = rate_of_containing_from_sample(samples=samples, level=level, 
                                                              estimate_of_true=args.central, 
                                                              true_val=b_centrals, ci_type="bayesian", 
                                                              bootstrap_repeats=100)
            rate *= 100
            rate_error *= 100

            b_rates.append(rate)
            b_rate_errors.append(rate_error)

        # error bars to be one standard error
        b_rate_errors = [e/2. for e in b_rate_errors]

        # saving result
        pickle.dump({"LEVELS_PERCENT":LEVELS_PERCENT, "Params:": param, "b_rates": b_rates, "b_rate_errors": b_rate_errors},
                    open(f"{param_name}.pkl", "wb"))
    else:
        files = pickle.load(open(f"{param_name}.pkl", "rb"))
        b_rates = files['b_rates']
        b_rate_errors = files['b_rate_errors']

    plot_containing_rates([LEVELS_PERCENT], [b_rates],
                          observed_rate_errors=[b_rate_errors],
                          xlabel=XLABEL, ylabel=YLABEL,
                          xlimits=[0, 100], ylimits=[0, 100],
                          out=param_name)

print("DONE")
