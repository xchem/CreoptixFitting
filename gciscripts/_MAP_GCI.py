import jax
import jax.numpy as jnp
from jax import jit, vmap

import numpy as np

from _MAP import _uniform_pdf, _lognormal_pdf, _log_prior_sigma
from _MAP import _log_likelihood_normal
from _model_GCI import filter_experiment_data


def _map_update_by_prior(init_trace, priors, num_experiments):
    """ """
    trace = init_trace.copy()

    N = init_trace[next(iter(init_trace))].shape[0]
    trace.update(
        {
            key: jnp.ones(N) * priors[key]
            for key in ["logka_C", "logKd_C", "Rmax_C"]
            if key in priors
        }
    )

    if "alpha" in priors.keys():
        for idx in range(num_experiments):
            trace[f"alpha:{idx:02d}"] = jnp.ones(N) * priors["alpha"]
        (
            trace.update(
                {
                    "logka_C": jnp.zeros(N),
                    "logKd_C": jnp.zeros(N),
                    "Rmax_C": jnp.zeros(N),
                }
            )
            if priors.get("alpha") == 1
            else None
        )

    return trace


def _log_priors(mcmc_trace, experiments, nsamples=None):
    """
    Sum of log prior of all parameters, assuming they follows uniform distribution

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
    experiments     : list of dict containing information of experiments
    nsamples        : int, number of samples to find MAP
    ----------
    Return:
        An array which size equals to mcmc_trace[:samples], each position corresponds
        to sum of log prior calculated by values of parameters from mcmc_trace
    """
    params_name = []
    for name in mcmc_trace.keys():
        if not name.startswith("sigma") and not name.startswith("Rmax"):
            params_name.append(name)

    if nsamples is None:
        nsamples = len(mcmc_trace[params_name[0]])
    assert nsamples <= len(mcmc_trace[params_name[0]]), "nsamples too big"

    log_priors = jnp.zeros(nsamples)

    f_prior_ka = vmap(lambda param: _uniform_pdf(param, 4.5, 20.0))
    f_prior_kd = vmap(lambda param: _uniform_pdf(param, 0.0, 20.0))
    f_prior_Kd = vmap(lambda param: _uniform_pdf(param, -35.5, 0))
    f_prior_Rmax_C = vmap(lambda param: _uniform_pdf(param, 0.0, 500.0))
    f_prior_decay = vmap(lambda param: _uniform_pdf(param, 0.0, 0.5))
    f_prior_alpha = vmap(lambda param: _uniform_pdf(param, 0, 0.3))
    f_prior_epsilon = vmap(lambda param: _uniform_pdf(param, 0.0, 10.0))

    for key in mcmc_trace.keys():
        if key.startswith("logka"):
            log_priors += jnp.log(f_prior_ka(mcmc_trace[key][:nsamples]))
        if key.startswith("logkd"):
            log_priors += jnp.log(f_prior_kd(mcmc_trace[key][:nsamples]))
        if key.startswith("logKd"):
            log_priors += jnp.log(f_prior_Kd(mcmc_trace[key][:nsamples]))
        if key.startswith("Rmax_C"):
            log_priors += jnp.log(f_prior_Rmax_C(mcmc_trace[key][:nsamples]))
        if key.startswith("rate_decay"):
            log_priors += jnp.log(f_prior_decay(mcmc_trace[key][:nsamples]))
        if key.startswith("alpha"):
            log_priors += jnp.log(f_prior_alpha(mcmc_trace[key][:nsamples]))
        if key.startswith("epsilon"):
            log_priors += jnp.log(f_prior_epsilon(mcmc_trace[key][:nsamples]))

    if "conc" in mcmc_trace.keys():
        conc_uM = experiment["analyte_concentration"] * 1e6
        dE = 0.1
        log_priors += jnp.log(
            _lognormal_pdf(mcmc_trace["conc"][:nsamples], conc_uM, dE * conc_uM)
        )

    for idx, expt in enumerate(experiments):
        try:
            idx_expt = expt["index"]
        except:
            idx_expt = idx

        to_fit = expt["to_fit"]
        if f"log_sigma_Rt:{idx}" in mcmc_trace.keys():
            log_priors += _log_prior_sigma(
                mcmc_trace=mcmc_trace,
                data=expt["Rt"],
                sigma_name=f"log_sigma_Rt:{idx}",
                nsamples=nsamples,
            )

        if "Rt" in expt.keys():
            maxR = jnp.max(expt["Rt"])
            minR = jnp.min(expt["Rt"])

            if f"Rmax:{idx:02d}" in mcmc_trace.keys():
                f_prior_Rmax = vmap(
                    lambda param: _uniform_pdf(param, 0.2 * maxR, 10 * maxR)
                )
                log_priors += jnp.log(
                    f_prior_Rmax(mcmc_trace[f"Rmax:{idx:02d}"][:nsamples])
                )

            if f"Rmax_P:{idx:02d}" in mcmc_trace.keys():
                f_prior_Rmax = vmap(
                    lambda param: _uniform_pdf(param, 0.2 * maxR, 10 * maxR)
                )
                log_priors += jnp.log(
                    f_prior_Rmax(mcmc_trace[f"Rmax_P:{idx:02d}"][:nsamples])
                )

            if f"y_offset:{idx:02d}" in mcmc_trace.keys():
                f_prior_y_offset = vmap(
                    lambda param: _uniform_pdf(param, minR - 10, minR + 10.0)
                )
                log_priors += jnp.log(
                    f_prior_y_offset(mcmc_trace[f"y_offset:{idx:02d}"][:nsamples])
                )

    return np.array(log_priors)


def _log_likelihood_each_dataset(
    expt,
    trace_ka,
    trace_kd,
    trace_Rmax,
    trace_rate_decay,
    trace_y_offset,
    trace_conc,
    trace_sigma,
    nsamples,
):
    """
    Parameters:
    ----------
    expt            : list of multiple expts, each expt contain information about the data
    trace_ka        : trace of k_on
    trace_kd        : trace of k_off
    trace_Rmax      : trace of Rmax
    trace_y_offset  : trace of y_offset
    trace_conc      : trace of analyte concentration
    trace_rate_decay: trace of protein decay
    trace_sigma     : trace of experimental noise
    nsamples        : int, number of samples to find MAP
    ----------
    Return log likelihood given the experiment and mcmc_trace
    """
    log_likelihoods = jnp.zeros(nsamples, dtype=jnp.float64)

    Rt = expt["Rt"]
    to_fit = expt["to_fit"]
    ts = expt["ts"]
    cL = expt["cL_scale"]

    def f_dRdt(ka, kd, Rmax, rate_decay, y_offset, conc):
        R = Rt - y_offset
        Rmax_adj = jnp.exp(jnp.log(Rmax) + jnp.log(1.0 - rate_decay) * ts / 60)
        return ka * (cL * conc * 1e-6) * (Rmax_adj - R) - kd * R

    f_Rt = vmap(
        lambda ka, kd, Rmax, rate, y_offset, conc, sigma: _log_likelihood_normal(
            Rt[to_fit],
            jnp.array(
                jnp.cumsum(f_dRdt(ka, kd, Rmax, rate, y_offset, conc))
                * jnp.insert(jnp.diff(ts), 0, 0)
            )[to_fit],
            sigma,
        )
    )
    log_likelihoods += f_Rt(
        trace_ka,
        trace_kd,
        trace_Rmax,
        trace_rate_decay,
        trace_y_offset,
        trace_conc,
        trace_sigma,
    )

    return log_likelihoods


def _log_likelihoods(mcmc_trace, experiments, nsamples=None, show_progress=True):
    """
    Sum of log likelihood of all parameters given their distribution information in params_dist

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
    experiments     : list of dict
    nsamples        : int, number of samples to find MAP
    ----------
    Return:
        Sum of log likelihood given experiments, mcmc_trace, enzyme/ligand concentration uncertainty
    """
    params_name = []
    for name in mcmc_trace.keys():
        if not name.startswith("sigma"):
            params_name.append(name)

    if nsamples is None:
        nsamples = len(mcmc_trace[params_name[0]])
    assert nsamples <= len(mcmc_trace[params_name[0]]), "nsamples too big"

    log_likelihoods = jnp.zeros(nsamples)

    logka = mcmc_trace["logka"][:nsamples]
    ka = jnp.exp(logka)
    if "logKeq" in mcmc_trace.keys():
        logKeq = mcmc_trace["logKeq"][:nsamples]
        kd = jnp.exp(logka - logKeq)
    elif "logKd" in mcmc_trace.keys():
        logKd = mcmc_trace["logKd"][:nsamples]
        kd = jnp.exp(logka + logKd)
    else:
        kd = jnp.exp(mcmc_trace["logkd"][:nsamples])

    if "conc" in mcmc_trace.keys():
        conc_uM = mcmc_trace["conc"][:nsamples]
    else:
        conc_uM = None

    for idx, expt in enumerate(experiments):
        try:
            idx_expt = expt["index"]
        except:
            idx_expt = idx

        Rmax_key = f"Rmax:{idx:02d}"
        Rmax = mcmc_trace[Rmax_key][:nsamples]

        if f"rate_decay:{idx:02d}" in mcmc_trace.keys():
            rate_decay = mcmc_trace[f"rate_decay:{idx:02d}"][:nsamples]
        else:
            rate_decay = jnp.zeros(nsamples)

        y_offset_key = f"y_offset:{idx:02d}"
        if f"y_offset:{idx:02d}" in mcmc_trace.keys():
            y_offset = mcmc_trace[y_offset_key][:nsamples]
        else:
            y_offset = jnp.zeros(nsamples)

        if conc_uM is None:
            conc_uM = expt["analyte_concentration"] * 1e6 * jnp.ones(nsamples)

        if f"log_sigma_Rt:{idx:02d}" in mcmc_trace.keys():
            sigma = jnp.exp(mcmc_trace[f"log_sigma_Rt:{idx:02d}"][:nsamples])
        else:
            sigma = None

        log_likelihoods += _log_likelihood_each_dataset(
            expt=expt,
            trace_ka=ka,
            trace_kd=kd,
            trace_Rmax=Rmax,
            trace_rate_decay=rate_decay,
            trace_y_offset=y_offset,
            trace_conc=conc_uM,
            trace_sigma=sigma,
            nsamples=nsamples,
        )

    return np.array(log_likelihoods)


def _log_likelihood_each_dataset_complex(
    expt,
    trace_ka_P,
    trace_kd_P,
    trace_Rmax_P,
    trace_ka_C,
    trace_kd_C,
    trace_Rmax_C,
    trace_rate_decay,
    trace_alpha,
    trace_epsilon,
    trace_y_offset,
    trace_conc,
    trace_sigma,
    nsamples,
):
    """
    Parameters:
    ----------
    expt          : list of multiple expts, each expt contain information about the data
    trace_ka        : trace of k_on
    trace_kd        : trace of k_off
    trace_Rmax      : trace of Rmax
    trace_y_offset  : trace of y_offset
    trace_conc      : trace of analyte concentration
    trace_sigma     : trace of experimental noise
    nsamples        : int, number of samples to find MAP
    ----------
    Return log likelihood given the experiment and mcmc_trace
    """
    log_likelihoods = jnp.zeros(nsamples, dtype=jnp.float64)

    Rt = expt["Rt"]
    R_CL = expt["Rt_FC1"]
    cL = expt["cL_scale"]
    to_fit = expt["to_fit"]
    ts = expt["ts"]

    def f_dRdt_PL(ka_P, kd_P, Rmax_P, rate_decay, epsilon, y_offset, conc):
        R = Rt - y_offset
        Rmax_P_adj = jnp.exp(jnp.log(Rmax_P) + jnp.log(1.0 - rate_decay) * ts / 60)
        return ka_P * (cL * conc * 1e-6) * (Rmax_P_adj - R) - kd_P * R + epsilon

    def f_dRdt_CL(ka_C, kd_C, Rmax_C, conc):
        return ka_C * (cL * conc * 1e-6) * (Rmax_C - R_CL) - kd_C * R_CL

    f_Rt = vmap(
        lambda ka_P, kd_P, Rmax_P, ka_C, kd_C, Rmax_C, rate, alpha, epsilon, y_offset, conc, sigma: _log_likelihood_normal(
            Rt[to_fit],
            jnp.array(
                jnp.cumsum(f_dRdt_PL(ka_P, kd_P, Rmax_P, rate, epsilon, y_offset, conc))
                * jnp.insert(jnp.diff(ts), 0, 0)
            )[to_fit]
            + alpha
            * R_CL[
                to_fit
            ],  # + alpha*jnp.array(jnp.cumsum(f_dRdt_CL(ka_C, kd_C, Rmax_C, conc))*jnp.insert(jnp.diff(ts),0,0))[to_fit],
            sigma,
        )
    )
    log_likelihoods += f_Rt(
        trace_ka_P,
        trace_kd_P,
        trace_Rmax_P,
        trace_ka_C,
        trace_kd_C,
        trace_Rmax_C,
        trace_rate_decay,
        trace_alpha,
        trace_epsilon,
        trace_y_offset,
        trace_conc,
        trace_sigma,
    )

    return log_likelihoods


def _log_likelihoods_complex(
    mcmc_trace, experiments, nsamples=None, show_progress=True
):
    """
    Sum of log likelihood of all parameters given their distribution information in params_dist

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
    experiments     : list of dict
    nsamples        : int, number of samples to find MAP
    ----------
    Return:
        Sum of log likelihood given experiments and mcmc_trace
    """
    params_name = []
    for name in mcmc_trace.keys():
        if not name.startswith("sigma"):
            params_name.append(name)

    if nsamples is None:
        nsamples = len(mcmc_trace[params_name[0]])
    assert nsamples <= len(mcmc_trace[params_name[0]]), "nsamples too big"

    log_likelihoods = jnp.zeros(nsamples)

    # Extracting mcmc_trace from global parameters
    logka_P = mcmc_trace["logka_P"][:nsamples]
    ka_P = jnp.exp(logka_P)
    if "logkd_P" in mcmc_trace.keys():
        logkd_P = mcmc_trace["logkd_P"][:nsamples]
        kd_P = jnp.exp(logkd_P)
    else:
        logKd_P = mcmc_trace["logKd_P"][:nsamples]
        kd_P = jnp.exp(logka_P + logKd_P)

    logka_C = mcmc_trace["logka_C"][:nsamples]
    ka_C = jnp.exp(logka_C)
    if "logkd_C" in mcmc_trace.keys():
        logkd_C = mcmc_trace["logkd_C"][:nsamples]
        kd_C = jnp.exp(logkd_C)
    else:
        logKd_C = mcmc_trace["logKd_C"][:nsamples]
        kd_C = jnp.exp(logka_C + logKd_C)

    Rmax_C = mcmc_trace["Rmax_C"][:nsamples]

    if "conc" in mcmc_trace.keys():
        conc_uM = mcmc_trace["conc"][:nsamples]
    else:
        conc_uM = None

    # Filter out non-JAX-compatible data from each experiment
    filtered_experiments = [filter_experiment_data(exp) for exp in experiments]

    for idx, expt in enumerate(experiments):
        try:
            idx_expt = expt["index"]
        except:
            idx_expt = idx

        if f"rate_decay:{idx:02d}" in mcmc_trace.keys():
            rate_decay = mcmc_trace[f"rate_decay:{idx:02d}"][:nsamples]
        else:
            rate_decay = jnp.zeros(nsamples)

        # Extracting mcmc_trace from local paparameters
        if expt["binding_type"] == "specific":
            Rmax_P_key = f"Rmax_P:{idx:02d}"
            Rmax_P = mcmc_trace[Rmax_P_key][:nsamples]
            alpha = mcmc_trace[f"alpha:{idx:02d}"][:nsamples]

            if f"epsilon:{idx:02d}" in mcmc_trace.keys():
                epsilon = mcmc_trace[f"epsilon:{idx:02d}"][:nsamples]
            else:
                epsilon = jnp.zeros(nsamples)

        y_offset_key = f"y_offset:{idx:02d}"
        if f"y_offset:{idx:02d}" in mcmc_trace.keys():
            y_offset = mcmc_trace[y_offset_key][:nsamples]
        else:
            y_offset = jnp.zeros(nsamples)

        if conc_uM is None:
            conc_uM = expt["analyte_concentration"] * 1e6 * jnp.ones(nsamples)

        if f"log_sigma_Rt:{idx:02d}" in mcmc_trace.keys():
            sigma = jnp.exp(mcmc_trace[f"log_sigma_Rt:{idx:02d}"][:nsamples])
        else:
            sigma = None

        if sigma is not None:
            if expt["binding_type"] == "specific":
                log_likelihoods += _log_likelihood_each_dataset_complex(
                    expt=filtered_experiments[idx],
                    trace_ka_P=ka_P,
                    trace_kd_P=kd_P,
                    trace_Rmax_P=Rmax_P,
                    trace_ka_C=ka_C,
                    trace_kd_C=kd_C,
                    trace_Rmax_C=Rmax_C,
                    trace_rate_decay=rate_decay,
                    trace_alpha=alpha,
                    trace_epsilon=epsilon,
                    trace_y_offset=y_offset,
                    trace_conc=conc_uM,
                    trace_sigma=sigma,
                    nsamples=nsamples,
                )
            else:
                log_likelihoods += _log_likelihood_each_dataset(
                    expt=filtered_experiments[idx],
                    trace_ka=ka_C,
                    trace_kd=kd_C,
                    trace_Rmax=Rmax_C,
                    trace_rate_decay=rate_decay,
                    trace_y_offset=y_offset,
                    trace_conc=conc_uM,
                    trace_sigma=sigma,
                    nsamples=nsamples,
                )

    return np.array(log_likelihoods)


def map_finding(
    mcmc_trace, experiments, fitting_complex=False, nsamples=None, show_progress=True
):
    """
    Evaluate probability of a parameter set using posterior distribution
    Finding MAP (maximum a posterior) given prior distributions of parameters information

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling trace (group_by_chain=False)
    experiments     : list of dict
    fitting_complex     : boolean, if True, using non-specific binding model
    nsamples        : int, number of samples to find MAP
    ----------
    Return          : values of parameters that maximize the posterior
    """
    if show_progress:
        print("Calculing log of priors.")
    log_priors = _log_priors(
        mcmc_trace=mcmc_trace, experiments=experiments, nsamples=nsamples
    )

    if show_progress:
        print("Calculing log likelihoods.")

    if fitting_complex:
        f_log_likelihoods = _log_likelihoods_complex
    else:
        f_log_likelihoods = _log_likelihoods

    log_likelihoods = f_log_likelihoods(
        mcmc_trace=mcmc_trace,
        experiments=experiments,
        nsamples=nsamples,
        show_progress=show_progress,
    )

    log_probs = log_priors + log_likelihoods
    map_idx = np.nanargmax(log_probs)
    if show_progress:
        print("Map index: %d" % map_idx)

    map_params = {}
    for name in mcmc_trace.keys():
        map_params[name] = mcmc_trace[name][map_idx]

    return [map_idx, map_params, log_probs]
