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

1. Setting up the main directory

```bash
    DIR='/home/vla/python/gci'
```

2. Install the packages and download the GitHub repository:

```bash
    git clone 'https://github.com/vanngocthuyla/gci.git'
```

3. Global Fitting for GCI Experiment

Assuming `DIR` is your working directory, you can perform global fitting for one set of GCI experiments, including the subtracted datasets (e.g., FC2-1, FC3-1, and FC4-1) from cycle 13, by running the following commands:  

```bash

mkdir $DIR/example/output
python $DIR/scripts/run_fitting_GCI.py \
    --out_dir $DIR/example/output/13_Y \
    --global_fitting --fitting_subtract --return_y_offset \
    --init_niters 1000 --init_nburn 200 \
    --niters 5000 --nburn 2000 --nchain 4 --random_key 0 \
    --analyte_file $DIR/example/input/ZIKV_Subtraction.csv \
    --analyte_keys_included "Fc=2-1-13_Y Fc=3-1-13_Y Fc=4-1-13_Y" \
    --analyte_concentration_uM 10.0 \
    --calibration_file $DIR/example/input/ZIKV_DMSO.csv \
    --calibration_keys_included "Fc=2-11_Y Fc=3-11_Y Fc=4-11_Y" \
    --end_dissociation 10.0
```

**Arguments for `run_fitting_GCI.py`**:  

- **`--out_dir`**: Specifies the directory where the results will be saved.  
- **`--global_fitting`**: If set to `True`, performs global fitting for subtracted FCs. If `False`, fits each subtracted FC separately.  
- **`--fitting_complex`**: If set to `True`, fits a complex model. Otherwise, fits a simple model.  
- **`--fitting_subtract`**: If set to `True`, the provided data is subtracted (e.g., `Fc=2-1-13`). If `False`, the provided data is raw (e.g., `Fc=2-13`).  
- **`--return_y_offset`**: If set to `True`, estimates the `y_offset`.  
- **`--init_niters`, `--init_nburn`, `--niters`, `--nburn`, `--nchain`**:  
  Define the number of MCMC samples and burn-in steps:  
  * `--init_niters`: Initial MCMC samples for tuning.  
  * `--init_nburn`: Burn-in samples during initial tuning.  
  * `--niters`: Total number of MCMC samples collected.  
  * `--nburn`: Burn-in samples during collection.  
  * `--nchain`: Number of chains for Bayesian regression.  
- **`--random_key`**: Sets the random key for Bayesian regression.  
- **`--analyte_file`**: Specifies the input data file.  
- **`--analyte_keys_included`**: Specifies the columns of analyzed data.  
- **`--analyte_keys_included_FC1`**: Required if `--fitting_subtract=False`.  
- **`--analyte_concentration_uM`**: Specifies the analyte concentration in µM.  
- **`--calibration_file`**: Specifies the DMSO calibration file.  
- **`--calibration_keys_included`**: Specifies the columns for DMSO data.  
- **`--end_dissociation`**: Defines the end time point for dissociation.  

Alternatively, you can adjust the script `/scripts/submit_fitting_GCI.py` to submit the fitting jobs directly.  

## SLURM job submission

### SLURM Job Information  
More details about SLURM jobs can be found via [this link](https://slurm.schedmd.com/overview.html).  

### Submitting a Job for the Example Dataset

To submit a job for the example dataset, follow these steps:  

1. **Edit the main directory**:  

Update the main directory in the file [main_dir.txt (https://github.com/vanngocthuyla/gci/blob/main/main_dir.txt).  

2. **Adjust the partition and Conda environment**:  

Modify the partition and Conda environment settings in the script [submit_fitting_GCI.py](https://github.com/vanngocthuyla/gci/blob/0fe27b22cf34e38131c5ddf285bc964b197c5f9f/scripts/submit_fitting_GCI.py#L167C1-L176C22).  

4. **Submit the job**:  

Run the following command to submit the job:  

   ```bash  
   bash run_me_submit_fitting_subtract.sh
   ```
5. **Job Automation and Output Details**

The code automatically detects datasets from different cycles in the input file `ZIKV.csv` and submits multiple jobs to the server. Each job is named according to its corresponding cycle number.

Output Information:

- All submission details are stored in `.job` and `.log` files located in the defined output directory.  
- Results for each experiment fitting are saved in subfolders named after their respective cycle numbers. Each subfolder includes:  
  * Autocorrelation plot
  * Trace plot
  * MCMC samples saved as `traces.pickle`
  * Summarized MCMC samples saved as `Summary.csv`
  * MAP estimates saved as `map.pickle` and `map.csv`
  * A plot of the fitted sensorgram (gray curve) overlaid with observed dissociation segment data (red dots), along with the mean and standard deviations of parameter estimates from Bayesian regression
  * A diagnostic plot for derivative and integral curves
