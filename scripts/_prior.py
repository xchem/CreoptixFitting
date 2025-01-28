from _prior_distribution import *

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _prior_GCI(experiments, args=None):
    """
    Parameters:
    ----------
    experiments     : list of dict, each containing information of a dataset
    return_conc     : boolean, if True, treating the analyte concentration as a parameter
    return_y_offset : boolean, if True, estimating y_offset
    ----------
    Returns a dictionary containing prior information.
    """
    params = {}

    # Global parameters
    params["logKd"] = uniform_prior(name="logKd", lower=-35.5, upper=0.0)
    params["logka"] = uniform_prior(name="logka", lower=4.5, upper=20.0)

    # Local parameters
    num_experiments = len(experiments)
    minR = jnp.zeros(num_experiments)
    maxR = jnp.zeros(num_experiments)

    # Initialize arrays for local parameters
    Rmax_array = jnp.zeros(num_experiments)
    protein_decay_array = jnp.zeros(num_experiments)
    y_offset_array = jnp.zeros(num_experiments)
    log_sigma_Rt_array = jnp.zeros(num_experiments)

    for idx, experiment in enumerate(experiments):
        minR = minR.at[idx].set(jnp.min(experiment["Rt"]))
        maxR = maxR.at[idx].set(jnp.max(experiment["Rt"]))

        # Sample Rmax as a local parameter
        Rmax_array = Rmax_array.at[idx].set(
            uniform_prior(
                name=f"Rmax:{idx:02d}", lower=0.2 * maxR[idx], upper=100 * maxR[idx]
            )
        )

        # Sample y_offset as a local parameter if specified
        if args.return_y_offset:
            y_offset_array = y_offset_array.at[idx].set(
                uniform_prior(
                    name=f"y_offset:{idx:02d}",
                    lower=minR[idx] - 10,
                    upper=minR[idx] + 10,
                )
            )
        else:
            y_offset_array = y_offset_array.at[idx].set(0.0)

        if args.include_protein_decay:
            protein_decay_array = protein_decay_array.at[idx].set(
                uniform_prior(name=f"rate_decay:{idx:02d}", lower=0.0, upper=0.5)
            )

        # Sample y_offset as a local parameter
        log_sigma_min, log_sigma_max = logsigma_guesses(experiment["Rt"])
        log_sigma_Rt_array = log_sigma_Rt_array.at[idx].set(
            uniform_prior(
                name=f"log_sigma_Rt:{idx:02d}", lower=log_sigma_min, upper=log_sigma_max
            )
        )

    # Store local parameters in the params dictionary
    params["Rmax"] = Rmax_array
    params["rate_decay"] = protein_decay_array
    params["y_offset"] = y_offset_array
    params["log_sigma_Rt"] = log_sigma_Rt_array

    # Sample concentration if needed
    if args.return_conc:
        conc_uM = (
            experiments[0]["analyte_concentration"] * 1e6
        )  # Assuming the same concentration for simplicity
        dE = 0.1
        params["conc"] = lognormal_prior("conc", conc_uM, dE * conc_uM)

    return params


def _prior_each_data(all_params, idx):
    """
    Extracting params by idx from the dictionary of all parameters
    """
    params = {
        "logka": all_params["logka"],
        "logKd": all_params["logKd"],
        "Rmax": all_params["Rmax"][idx],
        "rate_decay": all_params["rate_decay"][idx],
        "y_offset": all_params["y_offset"][idx],
        "log_sigma_Rt": all_params["log_sigma_Rt"][idx],
    }
    if "conc" in all_params.keys():
        params["conc"] = all_params["conc"]
    return params


def _prior_GCI_complex(experiments, priors={}, args=None):
    """
    Parameters:
    ----------
    experiments     : list of dict, each containing information of a dataset
    return_conc     : boolean, if True, treating the analyte concentration as a parameter
    return_y_offset : boolean, if True, estimating y_offset
    ----------
    Returns a dictionary containing prior information for specific/non-specific
    """
    params = {}

    # Global parameters
    params["logKd_P"] = uniform_prior(name="logKd_P", lower=-35.5, upper=0.0)
    params["logka_P"] = uniform_prior(name="logka_P", lower=4.5, upper=20.0)

    if not "logKd_C" in priors.keys():
        params["logKd_C"] = uniform_prior(name="logKd_C", lower=-35.5, upper=0.0)
    else:
        params["logKd_C"] = priors["logKd_C"]

    if not "logka_C" in priors.keys():
        params["logka_C"] = uniform_prior(name="logka_C", lower=4.5, upper=20.0)
    else:
        params["logka_C"] = priors["logka_C"]

    if not "Rmax_C" in priors.keys():
        params["Rmax_C"] = uniform_prior(name="Rmax_C", lower=0, upper=500.0)
    else:
        params["Rmax_C"] = priors["Rmax_C"]

    # Local parameters
    num_experiments = len(experiments)
    minR = jnp.zeros(num_experiments)
    maxR = jnp.zeros(num_experiments)

    # Initialize arrays for local parameters
    Rmax_array = jnp.zeros(num_experiments)
    protein_decay_array = jnp.zeros(num_experiments)
    epsilon_array = jnp.zeros(num_experiments)
    y_offset_array = jnp.zeros(num_experiments)
    log_sigma_Rt_array = jnp.zeros(num_experiments)

    if "alpha" in priors.keys():
        alpha_array = jnp.ones(num_experiments) * priors["alpha"]
    else:
        alpha_array = jnp.zeros(num_experiments)

    for idx, experiment in enumerate(experiments):
        minR = minR.at[idx].set(jnp.min(experiment["Rt"]))
        maxR = maxR.at[idx].set(jnp.max(experiment["Rt"]))

        if experiment["binding_type"] == "specific":
            # Sample Rmax as a local parameter
            Rmax_array = Rmax_array.at[idx].set(
                uniform_prior(
                    name=f"Rmax_P:{idx:02d}",
                    lower=0.2 * maxR[idx],
                    upper=10 * maxR[idx],
                )
            )

            # Sample alpha as a local parameter if not specified
            if not "alpha" in priors.keys():
                alpha_array = alpha_array.at[idx].set(
                    uniform_prior(name=f"alpha:{idx:02d}", lower=0.0, upper=0.3)
                )

            # Sample epsilon as a local parameter
            if args.return_epsilon:
                epsilon_array = epsilon_array.at[idx].set(
                    uniform_prior(name=f"epsilon:{idx:02d}", lower=0.0, upper=10.0)
                )

        else:
            Rmax_array = Rmax_array.at[idx].set(jnp.nan)
            alpha_array = alpha_array.at[idx].set(jnp.nan)

        if args.include_protein_decay:
            protein_decay_array = protein_decay_array.at[idx].set(
                uniform_prior(name=f"rate_decay:{idx:02d}", lower=0.0, upper=0.5)
            )

        # Sample y_offset as a local parameter if specified
        if args.return_y_offset:
            y_offset_array = y_offset_array.at[idx].set(
                uniform_prior(
                    name=f"y_offset:{idx:02d}",
                    lower=minR[idx] - 10,
                    upper=minR[idx] + 20,
                )
            )

        # Sample log_sigma as a local parameter
        log_sigma_min, log_sigma_max = logsigma_guesses(experiment["Rt"])
        log_sigma_Rt_array = log_sigma_Rt_array.at[idx].set(
            uniform_prior(
                name=f"log_sigma_Rt:{idx:02d}", lower=log_sigma_min, upper=log_sigma_max
            )
        )

    # Store local parameters in the params dictionary
    params["Rmax_P"] = Rmax_array
    params["rate_decay"] = protein_decay_array
    params["alpha"] = alpha_array
    params["epsilon"] = epsilon_array
    params["y_offset"] = y_offset_array
    params["log_sigma_Rt"] = log_sigma_Rt_array

    # Sample concentration if needed
    if args.return_conc:
        conc_uM = (
            experiments[0]["analyte_concentration"] * 1e6
        )  # Assuming the same concentration for simplicity
        dE = 0.1
        params["conc"] = lognormal_prior("conc", conc_uM, dE * conc_uM)

    return params


def _prior_each_data_complex(all_params, idx, fitting_specific=False):
    """
    Extracting params by idx from the dictionary of all parameters for specific/non-specific model
    """
    if fitting_specific:
        params = {
            "logka_P": all_params["logka_P"],
            "logKd_P": all_params["logKd_P"],
            "logka_C": all_params["logka_C"],
            "logKd_C": all_params["logKd_C"],
            "Rmax_P": all_params["Rmax_P"][idx],
            "Rmax_C": all_params["Rmax_C"],
            "alpha": all_params["alpha"][idx],
            "epsilon": all_params["epsilon"][idx],
        }
    else:
        params = {
            "logka": all_params["logka_C"],
            "logKd": all_params["logKd_C"],
            "Rmax": all_params["Rmax_C"],
        }

    for key in ["rate_decay", "y_offset", "log_sigma_Rt"]:
        params[key] = all_params[key][idx]

    if "conc" in all_params.keys():
        params["conc"] = all_params["conc"]

    return params
