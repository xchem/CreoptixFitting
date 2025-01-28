import os
import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import random

# numpyro, a probabilistic programming language for Bayesian statistics
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform
from numpyro.infer import MCMC, NUTS, init_to_value

# For visualizing results
import arviz as az

# For saving trace
import pickle

from _model_GCI import _fitting_expts
from _MAP_GCI import map_finding, _map_update_by_prior
from _analysis import _report_params, _report_params_complex, _extract_params_by_idx
from _plot_GCI import (
    plot_Rt_dRdt,
    plot_Creoptix_Bayesian,
    plot_Rt_dRdt_complex,
    plot_Creoptix_Bayesian_complex,
)

jax.config.update("jax_enable_x64", True)


def fitting_expts(experiments, save_dir, args):
    """
    Parameters:
    ----------
    experiments     : list of dict of multiple experiments
        Each enzymes dataset contains multiple experimental datasets, including Rt, dRdt, ts, cL_scale (cLt), end dissociation time
    save_dir        : saving directory
    args            : class comprises other model arguments
        global_fitting  : boolean, if True, fitting globally
        return_conc     : boolean, if True, treating analyte concentration as parameter
        return_y_offset : boolean, if True, estimating y_offset for each sensogram
        fitting_complex : boolean, if True, using non-specific binding model
        init_niters     : int, the number of iterations for initial MCMC
        init_nburn      : int, the number of burn-in for initial MCMC
        niters          : int, the number of iterations for MCMC
        nburn           : int, the number of burn-in for MCMC
        nchain          : int, the number of MCMC chains
        random_key      : int, random key
        end_dissociation: float, the ended dissociation time
    ----------
    Return:
        Fitting the Bayesian model to estimate the kinetics parameters and noise of all datasets
        Saving all related results (correlation plot, trace plot, trace summary)
        Plotting MAP with observed data
    """
    rng_key = random.split(random.PRNGKey(args.random_key), args.nchain)

    # Extracting some prior information
    priors = {
        key: getattr(args, key)
        for key in ["logka_C", "logKd_C", "Rmax_C", "alpha"]
        if hasattr(args, key) and not jnp.isnan(getattr(args, key))
    }
    (
        priors.update({"logKd_C": 0.0, "logka_C": 0.0, "Rmax_C": 0.0})
        if "alpha" in priors and priors["alpha"] == 0.0
        else None
    )

    num_experiments = len(experiments)

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    if not os.path.isfile(os.path.join(save_dir, "traces.pickle")):
        kernel = NUTS(_fitting_expts)
        mcmc = MCMC(
            kernel,
            num_warmup=args.init_nburn,
            num_samples=args.init_niters,
            num_chains=args.nchain,
            progress_bar=False,
        )
        mcmc.run(rng_key, experiments=experiments, priors=priors, args=args)
        mcmc.print_summary()

        trace_init_group = mcmc.get_samples(group_by_chain=True)
        trace_init = mcmc.get_samples(group_by_chain=False)
        [map_index, map_params, log_probs] = map_finding(
            mcmc_trace=_map_update_by_prior(trace_init, priors, num_experiments),
            experiments=experiments,
            fitting_complex=args.fitting_complex,
        )

        init_values = {}
        for key in trace_init.keys():
            init_values[key] = trace_init[key][map_index]
        print(init_values)

        kernel = NUTS(_fitting_expts, init_strategy=init_to_value(values=init_values))
        mcmc = MCMC(
            kernel,
            num_warmup=args.nburn,
            num_samples=args.niters,
            num_chains=args.nchain,
            progress_bar=False,
        )
        mcmc.run(rng_key, experiments=experiments, priors=priors, args=args)
        mcmc.print_summary()

        print("Saving last state.")
        mcmc.post_warmup_state = mcmc.last_state
        pickle.dump(
            jax.device_get(mcmc.post_warmup_state),
            open(os.path.join(save_dir, "Last_state.pickle"), "wb"),
        )

        trace = mcmc.get_samples(group_by_chain=False)
        pickle.dump(trace, open(os.path.join(save_dir, "traces.pickle"), "wb"))

        az.plot_autocorr(trace)
        plt.savefig(os.path.join(save_dir, "Autocorrelation"))

        trace = mcmc.get_samples(group_by_chain=True)
        az.summary(trace).to_csv(os.path.join(save_dir, "Summary.csv"))

        data = az.convert_to_inference_data(trace)
        az.plot_trace(data, compact=False)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Plot_trace"))

        trace = mcmc.get_samples(group_by_chain=False)
    else:
        print("Loading MCMC file...")
        trace = pickle.load(open(os.path.join(save_dir, "traces.pickle"), "rb"))

    trace_map = _map_update_by_prior(trace, priors, num_experiments)

    [map_index, map_params, log_probs] = map_finding(
        mcmc_trace=trace_map,
        experiments=experiments,
        fitting_complex=args.fitting_complex,
    )

    with open(os.path.join(save_dir, "map.txt"), "w") as f:
        print("MAP index:" + str(map_index), file=f)
        print("Parameter value:", file=f)
        for key in trace.keys():
            print(key, ": %.3f" % trace[key][map_index], file=f)

    pickle.dump(log_probs, open(os.path.join(save_dir, "log_probs.pickle"), "wb"))

    map_values = {}
    for key in trace.keys():
        map_values[key] = trace[key][map_index]
    pickle.dump(map_values, open(os.path.join(save_dir, "map.pickle"), "wb"))

    if args.fitting_complex:
        f_report_params = _report_params_complex
        f_plot_Rt_dRdt = plot_Rt_dRdt_complex
        f_plot_Creoptix_Bayesian = plot_Creoptix_Bayesian_complex
    else:
        f_report_params = _report_params
        f_plot_Rt_dRdt = plot_Rt_dRdt
        f_plot_Creoptix_Bayesian = plot_Creoptix_Bayesian

    MAP, params_hat = f_report_params(
        trace_map,
        experiments,
        map_index=map_index,
        return_conc=args.return_conc,
        return_y_offset=args.return_y_offset,
    )

    if "conc" in MAP.keys():
        for experiment in experiments:
            experiment["adjusted_analyte_concentration"] = MAP["conc"]

    end_dissociation = args.end_dissociation
    for idx, experiment in enumerate(experiments):

        _MAP = _extract_params_by_idx(MAP, idx)
        _params_hat = _extract_params_by_idx(params_hat, idx)

        fig_name = str(experiment["keys_included"])

        if not args.fitting_complex:
            f_plot_Rt_dRdt(
                experiment,
                _MAP,
                end_dissociation,
                fig_name=f"Rt_{fig_name[3:6]}",
                OUTFILE=os.path.join(save_dir, f"dRdt_{fig_name[3:6]}"),
            )

            f_plot_Creoptix_Bayesian(
                experiment,
                _MAP,
                _params_hat,
                xlim=[0, end_dissociation],
                ylim=None,
                fig_name=f"Rt_{fig_name[3:6]}",
                OUTFILE=os.path.join(save_dir, f"Rt_{fig_name[3:6]}"),
            )

        else:
            f_plot_Rt_dRdt(
                experiment,
                _MAP,
                end_dissociation,
                fig_name=f"Rt_{fig_name[3:6]}",
                no_subtraction=False,
                OUTFILE=os.path.join(save_dir, f"dRdt_{fig_name[3:6]}"),
            )

            f_plot_Creoptix_Bayesian(
                experiment,
                _MAP,
                _params_hat,
                xlim=[0, end_dissociation],
                ylim=None,
                fig_name=f"Rt_{fig_name[3:6]}",
                no_subtraction=False,
                OUTFILE=os.path.join(save_dir, f"Rt_{fig_name[3:6]}"),
            )

            f_plot_Rt_dRdt(
                experiment,
                _MAP,
                end_dissociation,
                fig_name=f"Rt_{fig_name[3:4]}",
                no_subtraction=True,
                OUTFILE=os.path.join(save_dir, f"dRdt_{fig_name[3:4]}"),
            )

            f_plot_Creoptix_Bayesian(
                experiment,
                _MAP,
                _params_hat,
                xlim=[0, end_dissociation],
                ylim=None,
                fig_name=f"Rt_{fig_name[3:4]}",
                no_subtraction=True,
                OUTFILE=os.path.join(save_dir, f"Rt_{fig_name[3:4]}"),
            )

    return experiments, params_hat
