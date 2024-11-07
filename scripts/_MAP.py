import jax
import jax.numpy as jnp
from jax import jit, vmap

import numpyro
import numpyro.distributions as dist

from _prior_distribution import logsigma_guesses


def _log_prior_sigma(mcmc_trace, data, sigma_name, nsamples):
    """
    Sum of log prior of all log_sigma, assuming they follows uniform distribution

    Parameters:
    ----------
    mcmc_trace      : list of dict, trace of Bayesian sampling
    response        : list of jnp.array, data from each experiment
    nsamples        : int, number of samples to find MAP
    ----------

    """
    log_sigma_min, log_sigma_max = logsigma_guesses(data)
    f_log_prior_sigma = vmap(lambda sigma: _uniform_pdf(sigma, log_sigma_min, log_sigma_max))
    param_trace = mcmc_trace[sigma_name][: nsamples]
    return jnp.log(f_log_prior_sigma(param_trace))


def _log_likelihood_normal(response_actual, response_model, sigma):
    """
    PDF of log likelihood of normal distribution

    Parameters
    ----------
    response_actual : jnp.array, response of data
    response_model  : jnp.array, predicted data
    sigma           : standard deviation
    ----------
    Return:
        Sum of log PDF of response_actual given normal distribution N(response_model, sigma^2)
    """
    return jnp.nansum(dist.Normal(0, 1).log_prob((response_model - response_actual)/sigma))


def _uniform_pdf(x, lower, upper):
    """
    PDF of uniform distribution

    Parameters
    ----------
    x               : float
    lower           : float, lower of uniform distribution
    upper           : float, upper of uniform distribution
    ----------
    Return:
        PDF of x values given uniform distribution U(lower, upper)
    """
    # assert upper > lower, "upper must be greater than lower"
    # if (x < lower) or (x > upper):
    #     return 0.
    return 1./(upper - lower)


def _gaussian_pdf(x, mean, std):
    """
    PDF of gaussian distribution

    Parameters
    ----------
    x               : float
    mean            : float, mean/loc of gaussian distribution
    std             : float, standard deviation of gaussian distribution
    ----------
    Return:
        PDF of x values given gaussian distribution N(loc, scale)
    """
    return jnp.exp(dist.Normal(loc=0, scale=1).log_prob((x-mean)/std))


def _lognormal_pdf(x, stated_center, uncertainty):
    """
    PDF of lognormal distribution
    Ref: https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html

    Parameters
    ----------
    x               : float
    stated_center   : float, mean of normal distribution
    uncertainty     : float, scale of normal distribution
    ----------
    Return:
        PDF of x values given lognormal distribution from the normal(loc, scale)
    """

    # if x <= 0:
    #     return 0.
    # else:
    m = stated_center
    v = uncertainty**2

    mu = np.log(m / jnp.sqrt(1 + (v / (m ** 2))))
    sigma_2 = jnp.log(1 + (v / (m**2)))

    return 1 / x / jnp.sqrt(2 * jnp.pi * sigma_2) * jnp.exp(-0.5 / sigma_2 * (jnp.log(x) - mu)**2)
