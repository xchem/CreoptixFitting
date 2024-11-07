import jax
import jax.numpy as jnp
from jax import jit, vmap, lax

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform

from _prior import _prior_GCI, _prior_GCI_complex, _prior_each_data, _prior_each_data_complex

jax.config.update("jax_enable_x64", True)

@jax.jit
def _jax_Rt_model(Rt, ts, cLt, ka, kd, Rmax, rate_decay, y_offset):
    """
    Parameters:
    ----------
    Rt, ts, cLt    : arrays of receptor, time, and ligand concentrations
    ka, kd         : rate constants, v
    Rmax, y_offset : parameters for the model
    """
    Rt_update  = Rt - y_offset
    Rmax_adj   = jnp.exp(jnp.log(Rmax) + jnp.log(1. - rate_decay)*ts/60)
    dRdt_model = ka * cLt * (Rmax_adj - Rt_update) - kd*Rt_update
    return jnp.cumsum(dRdt_model) * jnp.insert(jnp.diff(ts), 0, 0)


@jax.jit
def _jax_Rt_model_complex(Rt, R_CL, cLt, ts, ka_P, kd_P, Rmax_P, 
                          ka_C, kd_C, Rmax_C, alpha, rate_decay, epsilon, y_offset):
    """
    Parameters:
    ----------
    Rt, ts, cLt    : arrays of response, time, and normalized analyte concentrations
    ka, kd         : rate constants
    Rmax, y_offset : other parameters for the model
    alpha          : float, fraction of chip available when protein is bound
    epsilon        : float, gradual noise
    rate_decay     : float, rate of protein degraded every minute
    """
    R_PL        = Rt - y_offset
    Rmax_P_adj  = jnp.exp(jnp.log(Rmax_P) + jnp.log(1. - rate_decay)*ts/60)
    dRdt_PL     = ka_P*cLt*(Rmax_P_adj-R_PL) - kd_P*R_PL + epsilon
    dRdt_CL     = ka_C*cLt*(Rmax_C-R_CL) - kd_C*R_CL
    R_PL_model  = jnp.cumsum(dRdt_PL) * jnp.insert(jnp.diff(ts), 0, 0)
    R_CL_model  = jnp.cumsum(dRdt_CL) * jnp.insert(jnp.diff(ts), 0, 0)
    return R_PL_model + alpha*R_CL #R_CL_model


def filter_experiment_data(experiment):
    """
    Filter out non-JAX-compatible data from the experiment dictionary.
    """
    experiment_jax = {'Rt': jnp.array(experiment['Rt']),
                      'ts': jnp.array(experiment['ts']),
                      'to_fit': jnp.array(experiment['to_fit']),
                      'idx_to_fit': jnp.array(experiment['to_fit']),
                      'analyte_concentration': jnp.array(experiment['analyte_concentration']),
                      'cL_scale': jnp.array(experiment['cL_scale'])}
    for key in ['Rt_FC1', 'cL_scale_FC1']:
        if key in experiment.keys():
            experiment_jax[key] = experiment[key]
    return experiment_jax


def convert_experiments_to_jax_format(experiments):
    """
    Converting experiments (list of dicts) into a dictionary of arrays
    """
    keys = experiments[0].keys()
    jax_compatible_experiments = {key: jnp.array([exp[key] for exp in experiments]) for key in keys}
    return jax_compatible_experiments


def _fitting_each_GCI_dataset_deterministic(experiment, params, return_conc=False, fitting_specific=False):
    """
    Process a single dataset, applying the correct model based on whether it's a simple or complex binding model.

    Parameters:
    ----------
    experiment       : dict containing experiment data.
    params           : dict containing parameters
    fitting_specific : bool, if True, uses the complex model; otherwise, uses the simple model
    return_conc      : bool, if True, calculates ligand concentration based on params['conc']
    
    """
    ts = jnp.array(experiment['ts'])
    if return_conc:
        cLt = experiment['cL_scale'] * params['conc'] * 1E-6
    else:
        cLt = experiment['cL_scale'] * experiment['analyte_concentration']

    if fitting_specific:
        # Using complex model
        Rt   = jnp.array(experiment['Rt'])
        R_CL = jnp.array(experiment['Rt_FC1'])
        
        ka_P = jnp.exp(params['logka_P'])
        kd_P = jnp.exp(params['logka_P'] + params['logKd_P'])
        ka_C = jnp.exp(params['logka_C'])
        kd_C = jnp.exp(params['logka_C'] + params['logKd_C'])
        
        Rt_model = _jax_Rt_model_complex(Rt=Rt, R_CL=R_CL, cLt=cLt, ts=ts,
                                         ka_P=ka_P, kd_P=kd_P, Rmax_P=params['Rmax_P'],
                                         ka_C=ka_C, kd_C=kd_C, Rmax_C=params['Rmax_C'],
                                         alpha=params['alpha'], rate_decay=params['rate_decay'], 
                                         epsilon=params['epsilon'], y_offset=params['y_offset'])
    else:
        # Using simple model
        Rt = jnp.array(experiment['Rt'])
        ka = jnp.exp(params['logka'])
        kd = jnp.exp(params['logka'] + params['logKd'])
        Rt_model = _jax_Rt_model(Rt=Rt, ts=ts, cLt=cLt, ka=ka, kd=kd, Rmax=params['Rmax'], 
                                 rate_decay=params['rate_decay'], y_offset=params['y_offset'])

    return Rt_model


def _fitting_expts(experiments, priors={}, args=None):
    
    if args.fitting_complex:
        _fitting_expts_complex(experiments=experiments, priors=priors, args=args)
    
    else:
        _fitting_expts_simple_binding(experiments=experiments, args=args)


def _fitting_expts_simple_binding(experiments, args=None): 
    """
    Fitting GCI 1:1 binding model
    """
    # Filter out non-JAX-compatible data from each experiment
    filtered_experiments = [filter_experiment_data(exp) for exp in experiments]

    all_params = _prior_GCI(experiments=experiments, args=args)

    # Convert to JAX-compatible format
    experiments_jax = convert_experiments_to_jax_format(filtered_experiments)

    @jax.jit  # Apply JIT compilation to the deterministic function
    def deterministic_step(experiment, idx):
        return _fitting_each_GCI_dataset_deterministic(
            experiment=experiment,
            params=_prior_each_data(all_params, idx),
            return_conc=args.return_conc,
        )

    # Compute the deterministic part using vmap with the jitted function
    Rt_models = jax.vmap(deterministic_step)(experiments_jax, jnp.arange(len(experiments)))

    # Vectorized calculation of Normal distribution
    Rt_model_to_fit = jax.vmap(lambda R, mask: jnp.where(mask, R, 0))(Rt_models, experiments_jax['to_fit'])
    Rt_to_fit = jax.vmap(lambda R, mask: jnp.where(mask, R, 0))(experiments_jax['Rt'], experiments_jax['to_fit'])

    # Extract log_sigma directly from all_params (precomputed in _prior_GCI)
    log_sigmas = all_params['log_sigma_Rt']

    # Expand log_sigmas to match the shape of the Rt_model arrays
    log_sigmas_expanded = jnp.expand_dims(log_sigmas, axis=-1)

    # Perform vectorized sampling for each experiment's observed data
    numpyro.sample('R_obs', dist.Normal(loc=Rt_model_to_fit, scale=jnp.exp(log_sigmas_expanded)), obs=Rt_to_fit)


def _fitting_expts_complex(experiments, priors={}, args=None):
    
    # Filter out non-JAX-compatible data from each experiment
    filtered_experiments = [filter_experiment_data(exp) for exp in experiments]
    print("Fitting complex model...")
    
    if len(priors)>0:
        print(priors)
    all_params = _prior_GCI_complex(experiments, priors=priors.copy(), args=args)

    # Extract log_sigma directly from all_params (precomputed in _prior_GCI)
    log_sigmas = all_params['log_sigma_Rt']

    # Determine fitting specific flags for each experiment
    fitting_specific = [expt['binding_type'] == 'specific' for expt in experiments]

    for idx, experiment in enumerate(experiments):
        
        Rt_model = _fitting_each_GCI_dataset_deterministic(
            experiment=experiment,
            params=_prior_each_data_complex(all_params, idx, fitting_specific[idx]), 
            return_conc=args.return_conc,
            fitting_specific=fitting_specific[idx]) 

        Rt_model_to_fit = jnp.where(experiment['to_fit'], Rt_model, 0)
        Rt_to_fit = jnp.where(experiment['to_fit'], experiment['Rt'], 0)

        # Perform sampling for each experiment's observed data
        numpyro.sample(f'R_obs{idx:02d}', dist.Normal(loc=Rt_model_to_fit, scale=jnp.exp(log_sigmas[idx])), obs=Rt_to_fit)