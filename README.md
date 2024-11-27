## Introduction

This GitHub is designed to fit Grating-Coupled Interferometry (GCI) data from Creoptix WAVEsystem.

## Set up the running environment

To run the Bayesian regression, we need some python packages. 

  * jax v0.4.33
  * jaxlib v0.4.33
  * numpyro v0.15.3
  * pickle-mixin >= v1.0.2
  * arviz >= 0.20.0


If higher versions of JAX, JAXlib, and numpyro are installed, we need to check whether x64 `jax.numpy` can be used by executing the following code without any errors:

    import jax
    jax.config.update("jax_enable_x64", True)

## Running test

Setting up the main directory
    
    DIR='/home/vla/python'

Install the packages and download the GitHub repository:
    
    git clone 'https://github.com/vanngocthuyla/gci.git'


## SLURM job submission

### SLURM Job Information  
More details about SLURM jobs can be found via [this link](https://slurm.schedmd.com/overview.html).  

### Submitting a Job for the Example Dataset

To submit a job for the example dataset, follow these steps:  

1. **Edit the main directory**:  
   Update the main directory in the file [main_dir.txt (https://github.com/vanngocthuyla/gci/blob/main/main_dir.txt).  

2. **Adjust the partition and Conda environment**:  
   Modify the partition and Conda environment settings in the script [submit_fitting_GCI.py](https://github.com/vanngocthuyla/gci/blob/0fe27b22cf34e38131c5ddf285bc964b197c5f9f/scripts/submit_fitting_GCI.py#L167C1-L176C22).  

3. **Submit the job**:  
   Run the following command to submit the job:  

   ```bash  
   bash run_me_submit_fitting_subtract.sh  
   ```

