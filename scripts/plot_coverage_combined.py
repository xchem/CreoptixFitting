import os
import glob
import argparse
import pickle

import numpy as np
import jax.numpy as jnp
import pandas as pd

from uncertainties import ufloat
from uncertainties.umath import *

import matplotlib
import matplotlib.pyplot as plt

from matplotlib import font_manager

font_dirs = ["/home/vla/python/fonts/arial"]
for font in font_manager.findSystemFonts(font_dirs):
    font_manager.fontManager.addfont(font)
matplotlib.rcParams["font.family"] = ["arial"]

from _confidence_intervals import (
    rate_of_containing_from_sample,
    rate_of_containing_from_means_stds,
)
from _plot_confidence_intervals import plot_containing_rates

parser = argparse.ArgumentParser()

parser.add_argument("--mcmc_dir", type=str, default="")
parser.add_argument("--out_dir", type=str, default="")
parser.add_argument("--nonlinear_fit_result_file", type=str, default="")

parser.add_argument("--fitting_complex", action="store_true", default=False)
parser.add_argument("--parameter", type=str, default="logka logkd logKd")

parser.add_argument("--central", type=str, default="median")

args = parser.parse_args()

assert args.central in ["mean", "median"], "wrong central"

MARKERS = ("<", "o", "v")
COLORS = ("b", "r", "c")

LEVELS_PERCENT = np.linspace(10.0, 95.0, num=18)
print("LEVELS_PERCENT", LEVELS_PERCENT)
LEVELS = LEVELS_PERCENT / 100.0

params = args.parameter.split()
XLABEL_rename = {
    "logka": "$lnk_a$",
    "logkd": "$lnk_d$",
    "logKd": "$lnK_d$",
    "logka_P": "$lnk_{a,P}$",
    "logkd_P": "$lnk_{d,P}$",
    "logKd_P": "$lnK_{d,P}$",
    "logka_C": "$lnk_{a,C}$",
    "logkd_C": "$lnk_{d,C}$",
    "logKd_C": "$lnK_{d,C}$",
    "ka": "$k_a$",
    "kd": "$k_d$",
    "Kd": "$K_d$",
}

if len(args.out_dir) > 0:
    os.chdir(args.out_dir)

if len(args.mcmc_dir) > 0:

    # bayesian cis
    b_rates_list = {}
    b_rate_errors_list = {}
    if os.path.isfile("Bayesian.pkl"):
        b_file = pickle.load(open("Bayesian.pkl", "rb"))
        if b_file["LEVELS_PERCENT"].all() == LEVELS_PERCENT.all():
            b_rates_list = b_file["b_rates"]
            b_rate_errors_list = b_file["b_rate_errors"]

    if len(b_rates_list) == 0:
        for param in params:
            print("\n", param)

            if args.fitting_complex:
                param = param + "_P"

            if param.startswith("logkd"):
                param_name = param + "_1"
            else:
                param_name = param

            if os.path.isfile(f"{param_name[3:]}.pkl"):
                b_file = pickle.load(open(f"{param_name[3:]}.pkl", "rb"))
                if b_file["LEVELS_PERCENT"].all() == LEVELS_PERCENT.all():
                    print(f"Loading {param_name[3:]}.pkl")
                    b_rates = b_file["b_rates"]
                    b_rate_errors = b_file["b_rate_errors"]

            else:
                mcmc_trace_files = glob.glob(os.path.join(args.mcmc_dir, "*"))
                experiment_names = [
                    os.path.basename(file)
                    for file in mcmc_trace_files
                    if os.path.isfile(os.path.join(file, "traces.pickle"))
                ]

                # Load all parameter values from MCMC traces
                if param == "logkd":
                    samples = [
                        jnp.exp(
                            pickle.load(
                                open(
                                    os.path.join(args.mcmc_dir, name, "traces.pickle"),
                                    "rb",
                                )
                            )["logka"]
                            + pickle.load(
                                open(
                                    os.path.join(args.mcmc_dir, name, "traces.pickle"),
                                    "rb",
                                )
                            )["logKd"]
                        )
                        for name in experiment_names
                    ]
                elif param == "logkd_P":
                    samples = [
                        jnp.exp(
                            pickle.load(
                                open(
                                    os.path.join(args.mcmc_dir, name, "traces.pickle"),
                                    "rb",
                                )
                            )["logka_P"]
                            + pickle.load(
                                open(
                                    os.path.join(args.mcmc_dir, name, "traces.pickle"),
                                    "rb",
                                )
                            )["logKd_P"]
                        )
                        for name in experiment_names
                    ]
                else:
                    samples = [
                        jnp.exp(
                            pickle.load(
                                open(
                                    os.path.join(args.mcmc_dir, name, "traces.pickle"),
                                    "rb",
                                )
                            )[param]
                        )
                        for name in experiment_names
                    ]

                if args.central == "median":
                    b_centrals = [np.median([np.median(sample) for sample in samples])]
                elif args.central == "mean":
                    b_centrals = [np.mean([np.mean(sample) for sample in samples])]

                b_rates = []
                b_rate_errors = []
                for level in LEVELS:
                    print(level)
                    rate, rate_error = rate_of_containing_from_sample(
                        samples=samples,
                        level=level,
                        estimate_of_true=args.central,
                        true_val=b_centrals,
                        ci_type="bayesian",
                        bootstrap_repeats=100,
                    )
                    rate *= 100
                    rate_error *= 100

                    b_rates.append(rate)
                    b_rate_errors.append(rate_error)

                # error bars to be one standard error
                b_rate_errors = [e / 2.0 for e in b_rate_errors]

                pickle.dump(
                    {
                        "LEVELS_PERCENT": LEVELS_PERCENT,
                        "b_rates": b_rates,
                        "b_rate_errors": b_rate_errors,
                    },
                    open(f"{param_name[3:]}.pkl", "wb"),
                )

            b_rates_list[param[3:]] = b_rates
            b_rate_errors_list[param[3:]] = b_rate_errors

        pickle.dump(
            {
                "LEVELS_PERCENT": LEVELS_PERCENT,
                "b_rates": b_rates_list,
                "b_rate_errors": b_rate_errors_list,
            },
            open("Bayesian.pkl", "wb"),
        )

    for new_key, key in zip(["ka", "kd", "Kd"], ["ka_P", "kd_P", "Kd_P"]):
        if key in b_rates_list.keys():
            b_rates_list[new_key] = b_rates_list[key]
        if key in b_rate_errors_list.keys():
            b_rate_errors_list[new_key] = b_rate_errors_list[key]


if len(args.nonlinear_fit_result_file) > 0:
    # nonlinear ls cis
    MLE_results = pd.read_csv(args.nonlinear_fit_result_file, index_col=0)

    for exper in MLE_results.index:
        if np.any(MLE_results.loc[exper].isnull()):
            raise Exception(exper + " is null")

    g_rate_list = {}
    for param, mean_col, percent_error_col in zip(
        ["ka", "kd", "Kd"],
        ["ka (M-1s-1)", "kd (s-1)", ""],
        ["ka error (%)", "kd error (%)", ""],
    ):

        if param == "Kd":
            means_ka = MLE_results["ka (M-1s-1)"]
            stds_ka = MLE_results["ka error (%)"] * means / 100.0
            means_kd = MLE_results["kd (s-1)"]
            stds_kd = MLE_results["kd error (%)"] * means / 100.0

            means = []
            stds = []
            for i in range(len(means_ka)):
                Kd = ufloat(means_kd.iloc[i], stds_kd.iloc[i]) / ufloat(
                    means_ka.iloc[i], stds_ka.iloc[i]
                )
                means.append(Kd.n)
                stds.append(Kd.s)
        else:
            means = MLE_results[mean_col]
            stds = MLE_results[percent_error_col] * means / 100.0

        if args.central == "median":
            centrals = np.median(means)
        elif args.central == "mean":
            centrals = np.mean(means)

        g_rates = []
        for level in LEVELS:
            g_rate = rate_of_containing_from_means_stds(
                means, stds, level, estimate_of_true=args.central, true_val=centrals
            )
            g_rate *= 100
            g_rates.append(g_rate)

        g_rate_list[param] = g_rates

    pickle.dump(
        {"LEVELS_PERCENT": LEVELS_PERCENT, "g_rates": g_rate_list},
        open("NLS.pkl", "wb"),
    )

# Plotting
plt.rcParams["figure.autolayout"] = True
fig, axes = plt.subplots(2, 2, figsize=(3.2 * 2, 3.2 * 2), sharex=False, sharey=False)
axes = axes.flatten()

for i, param in enumerate(["ka", "kd", "Kd"]):
    XLABEL = "Predicted " + XLABEL_rename[param]
    YLABEL = "Observed " + XLABEL_rename[param]
    if len(args.mcmc_dir) > 0:
        plot_containing_rates(
            [LEVELS_PERCENT],
            [b_rates_list[param]],
            observed_rate_errors=[b_rate_errors_list[param]],
            xlabel=XLABEL,
            ylabel=YLABEL,
            xlimits=[0, 100],
            ylimits=[0, 100],
            markers=MARKERS[0],
            ax=axes[i],
            colors=COLORS[0],
            label="Bayesian",
        )
        handles, labels = axes[i].get_legend_handles_labels()
    if len(args.nonlinear_fit_result_file) > 0:
        plot_containing_rates(
            [LEVELS_PERCENT],
            [g_rate_list[param]],
            observed_rate_errors=None,
            xlabel=XLABEL,
            ylabel=YLABEL,
            xlimits=[0, 100],
            ylimits=[0, 100],
            markers=MARKERS[1],
            ax=axes[i],
            colors=COLORS[1],
            label="NLS",
        )
        handles, labels = axes[i].get_legend_handles_labels()

    # axes[i].text(0.05, 0.8, XLABEL_rename[param], fontsize=14, transform=axes[i].transAxes, color='k')

by_label = dict(zip(labels, handles))
axes[3].set_visible(False)
legend = fig.legend(
    by_label.values(), by_label.keys(), loc="center right", bbox_to_anchor=(1.15, 0.6)
)
plt.savefig("Containing_Plot", bbox_extra_artists=[legend], bbox_inches="tight")
