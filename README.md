# Introduction

This GitHub is designed to fit Grating-Coupled Interferometry (GCI) data from Creoptix WAVEsystem.

# Set up the running environment

To run the Bayesian regression, we need some python packages. 

  * jax v0.4.33
  * jaxlib v0.4.33
  * numpyro v0.15.3
  * pickle-mixin >= v1.0.2
  * arviz >= 0.20.0


If higher versions of JAX, JAXlib, and numpyro are installed, we need to check whether x64 `jax.numpy` can be used by executing the following code without any errors:

    import jax
    jax.config.update("jax_enable_x64", True)

# Running test

Set your working directory:
    
    DIR='/home/vla/python'

Install the packages and download the GitHub repository:
    
    git clone 'https://github.com/vanngocthuyla/gci.git'
