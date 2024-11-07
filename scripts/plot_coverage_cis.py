import os
import glob
import pickle
import argparse

import pandas as pd
import numpy as np
import jax.numpy as jnp

from uncertainties import ufloat
from uncertainties.umath import *

import matplotlib
from matplotlib import font_manager
font_dirs = ['/home/vla/python/fonts/arial']
for font in font_manager.findSystemFonts(font_dirs):
    font_manager.fontManager.addfont(font)
matplotlib.rcParams['font.family'] = ['arial']

from _confidence_intervals import bayesian_credible_interval, gaussian_ci_from_mean_std
from _plot_confidence_intervals import plot_vertically_stacked_cis

parser = argparse.ArgumentParser()

parser.add_argument( "--mcmc_dir",                     		type=str,   			default="")
parser.add_argument( "--out_dir",                           type=str,               default="")
parser.add_argument( "--exclude_experiment",                type=str,   			default="")

parser.add_argument( "--nonlinear_fit_result_file",         type=str,               default="")

parser.add_argument( "--parameter",                         type=str,   			default="logka logkd logKd")
parser.add_argument( "--level",                             type=float, 			default=0.95)
parser.add_argument( "--central",                           type=str,   			default="median")
parser.add_argument( "--xlimits",                           type=str,   			default=None)

parser.add_argument( "--global_fitting",                  	action="store_true",    default=False)

args = parser.parse_args()

assert args.central in ["mean", "median"], "wrong central"

if len(args.out_dir)>0:
    os.chdir(args.out_dir)

print("ploting " + args.parameter)

TRACES_FILE = "traces.pickle"

if args.xlimits is None: xlimits = None
else: xlimits = [ float(s) for s in args.xlimits.split() ]
print("xlimits = ", xlimits)

params = args.parameter.split()
exclude_experiment = args.exclude_experiment.split()
XLABELS = {"logka": "$lnk_a$", "logkd": "$lnk_d$", "logKd": "$lnK_d$",
           "logka_P": "$lnk_{a,P}$", "logkd_P": "$lnk_{d,P}$", "logKd_P": "$lnK_{d,P}$",
           "logka_C": "$lnk_{a,C}$", "logkd_C": "$lnk_{d,C}$", "logKd_C": "$lnK_{d,C}$",
           "ka": "$k_a$", "kd": "$k_d$", "Kd": "$K_d$", "ka_P": "$k_{a,P}$", "kd_P": "$k_{d,P}$"}

if len(args.mcmc_dir)>0:

    for param in params:
        
        if param.startswith('logkd') or param.startswith('kd'): 
            param_name = param+'_1'
        else: 
            param_name = param
        
        mcmc_trace_files = glob.glob(os.path.join(args.mcmc_dir, '*'))
        experiment_names = [os.path.basename(file) for file in mcmc_trace_files if os.path.isfile(os.path.join(file, "traces.pickle"))]

        experiment_names_2 = [name for name in experiment_names if name.startswith("2") and not name in exclude_experiment]
        experiment_names_3 = [name for name in experiment_names if name.startswith("3") and not name in exclude_experiment]
        experiment_names_4 = [name for name in experiment_names if name.startswith("4") and not name in exclude_experiment]

        if param.startswith('log'):
            if param == 'logkd':
                samples = {name: pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logka']+pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logKd'] for name in experiment_names}
            elif param == 'logkd_P':
                samples = {name: pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logka_P']+pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logKd_P'] for name in experiment_names}
            elif param == 'logkd_C':
                samples = {name: pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logka_C']+pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logKd_C'] for name in experiment_names}
            else:
                samples = {name: pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))[param] for name in experiment_names}
        else:
            if param == 'kd':
                samples = {name: jnp.exp(pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logka']+pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logKd']) for name in experiment_names}
            elif param == 'kd_P':
                samples = {name: jnp.exp(pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logka_P']+pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logKd_P']) for name in experiment_names}
            elif param == 'kd_C':
                samples = {name: jnp.exp(pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logka_C']+pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['logKd_C']) for name in experiment_names}
            else:
                samples = {name: jnp.exp(pickle.load(open(os.path.join(args.mcmc_dir, name, "traces.pickle"), "rb"))['log'+param]) for name in experiment_names}

        if not os.path.isfile(f'{param_name}_bic.pickle'):
            lowers          = []
            uppers          = []
            lower_errors    = []
            upper_errors    = []
            for exper in samples.keys():
                lower, upper, lower_error, upper_error = bayesian_credible_interval(samples[exper], args.level, bootstrap_repeats=1000)

                lowers.append(lower)
                uppers.append(upper)
                lower_errors.append(lower_error)
                upper_errors.append(upper_error)

            cis = {'lowers': lowers, 'uppers': uppers, 'lower_errors': lower_errors, 'upper_errors': upper_errors}
            pickle.dump(cis, open(f'{param_name}_bic.pickle', "wb"))
        else:
            cis = pickle.load(open(f'{param_name}_bic.pickle', "rb"))
            lowers = cis['lowers']
            uppers = cis['uppers']
            lower_errors = cis['lower_errors']
            upper_errors = cis['upper_errors']

        exclude_idx = np.array([np.where(np.array(experiment_names)==expt) for expt in exclude_experiment]).flatten()
        
        if len(exclude_idx)>0:
            lowers = [lowers[i] for i in range(len(lowers)) if not i in exclude_idx]
            uppers = [uppers[i] for i in range(len(uppers)) if not i in exclude_idx]
            lower_errors = [lower_errors[i] for i in range(len(lower_errors)) if not i in exclude_idx]
            upper_errors = [upper_errors[i] for i in range(len(upper_errors)) if not i in exclude_idx]

        if args.global_fitting:
            if args.central == "median":
                b_centrals = [ np.median( [np.median(sample) for sample in samples.values()] ) ]
            elif args.central == "mean":
                b_centrals = [ np.mean( [ np.mean(sample) for sample in samples.values() ] ) ]
        else:
            if args.central == "median":
                b_centrals =  [ np.median( [ np.median(samples[exper]) for exper in experiment_names_2 if exper in samples.keys()] ) ]
                b_centrals += [ np.median( [ np.median(samples[exper]) for exper in experiment_names_3 if exper in samples.keys()] ) ]
                b_centrals += [ np.median( [ np.median(samples[exper]) for exper in experiment_names_4 if exper in samples.keys()] ) ]
            elif args.central == "mean":
                b_centrals =  [ np.mean( [ np.mean(samples[exper]) for exper in experiment_names_2 if exper in samples.keys()] ) ]
                b_centrals += [ np.mean( [ np.mean(samples[exper]) for exper in experiment_names_3 if exper in samples.keys()] ) ]
                b_centrals += [ np.mean( [ np.mean(samples[exper]) for exper in experiment_names_4 if exper in samples.keys()] ) ]

        out = param_name + "_bic.pdf"

        # error bars to be one standard error
        lower_errors = [l/2. for l in lower_errors]
        upper_errors = [u/2. for u in upper_errors]

        plot_vertically_stacked_cis(lowers, uppers, XLABELS[param], out,
                                    lower_errors=lower_errors, upper_errors=upper_errors,
                                    centrals=b_centrals, xlimits=xlimits)

if len(args.nonlinear_fit_result_file)>0:

    MLE_results = pd.read_csv(args.nonlinear_fit_result_file, index_col=0)

    for exper in MLE_results.index:
        if np.any(MLE_results.loc[exper].isnull()):
            raise Exception(exper + " is null")

    for param, mean_col, percent_error_col in zip(['ka', 'kd', 'Kd'], ['ka (M-1s-1)', 'kd (s-1)', ""], ['ka error (%)', 'kd error (%)', ""]):

        if param.startswith('logkd') or param.startswith('kd'): 
            param_name = param+'_1'
        else: 
            param_name = param

        if param == 'Kd':
            means_ka = MLE_results['ka (M-1s-1)']
            stds_ka  = MLE_results['ka error (%)']*means/100.
            means_kd = MLE_results['kd (s-1)']
            stds_kd  = MLE_results['kd error (%)']*means/100.

            means = []
            stds = []
            for i in range(len(means_ka)):
                Kd = ufloat(means_kd.iloc[i], stds_kd.iloc[i])/ufloat(means_ka.iloc[i], stds_ka.iloc[i])
                means.append(Kd.n)
                stds.append(Kd.s)
        else:
            means = MLE_results[mean_col]
            stds  = MLE_results[percent_error_col]*means/100.
        
        if args.central == "median":
            centrals = [np.median(means)]
        elif args.central == "mean":
            centrals = [np.mean(means)]

        g_lowers  = []
        g_uppers  = []

        for mu, sigma in zip(means, stds):
            l, u = gaussian_ci_from_mean_std(mu, sigma, args.level)
            g_lowers.append(l)
            g_uppers.append(u)

        out = param_name + "_nls_cis"

        plot_vertically_stacked_cis(g_lowers, g_uppers, XLABELS[param], out,
                                    centrals=centrals, xlimits=xlimits)