# Global Binary Modelling with ELLC

```global_ellc.py``` is a generalised multi-instrument version of James Blake's emcee_combined.py

# Motivation

This code was written to model eclipsing MS+BD binaries and so contains
flexibility on a subset of binary parameters. More flexability can be
included easily later.

# Usage

To run the code you must specify a configuration file which describes the
data location, mcmc setup and initial parameters + priors etc. See below
for the example config file for J234318.41 (eclipsing BD):
Lines starting with # are ignored (comments). The comments mingled with
the configurations below explain how things are set up.

```sh
python global_ellc.py /path/to/config/file
```

Example config file:

```sh
###############################
######  DATA DESCRIPTION ######
###############################
# location
data_dir /Users/jmcc/Dropbox/EBLMs/J23431841
out_dir /Users/jmcc/Dropbox/EBLMs/J23431841/output
# data files, formats expected are either:
# lc filter filename time_col_id flux_col_id flux_err_col_id
# rv instrument filename time_col_id rv_col_id rverr_col_id
# col_ids are 0 indexed
lc Clear NITES_J234318.41_Clear_20120829_F1_A14.lc.txt 2 3 4
lc Clear NITES_J234318.41_Clear_20130923_F2_A14.lc.txt 2 3 4
lc Clear NITES_J234318.41_Clear_20131010_F1_A14.lc.txt 2 3 4
lc Clear NITES_J234318.41_Clear_20141001_F1_A14.lc.txt 2 3 4
rv FIES J234318.41_NOT.rv 0 1 2
rv SOPHIE J234318.41_SOPHIE.rv 0 1 2
rv PARAS J234318.41_PARAS.rv 0 1 2
###############################
######  MCMC DESCRIPTION ######
###############################
# number of MCMC steps
nsteps 10
# scale MCMC nwalkers
# nwalkers = walker_scaling * 4*n_parameters
walker_scaling 1
# MCMC parameters can be either:
#   Fixed parameters:
#        parameter_name F value
#   Uniform prior parameters:
#       parameter_name U seed_value weight prior_l prior_h
#   No prior parameters:
#       parameter_name N seed_value weight
# walkers are normally distributed around (seed_value, weights)
sbratio F 0.0
r1_a U 0.1 0.0001 0.02 0.3
r2_r1 U 0.08 0.0001 0.03 0.15
incl U 89.6232 0.01 88.0 90.0
t0 N 2453592.74192 0.001
period N 16.9535452 0.0001
ecc U 0.16035 0.001 0.1 0.2
omega U 78.39513 0.1 70.0 90.0
ldc_1_1 F 0.3897
ldc_1_2 F 0.1477
q U 0.09649 0.001 0.05 0.145
K U 10.0 5.0 15.0
# systemic velocity parameters are instrument specific
# they are best kept to the end of the config file and defined as:
#   vsys U instrument seed_value weight prior_l prior_h
vsys U FIES -21.133 0.1 -25.0 -15.0
vsys U SOPHIE -21.122 0.1 -25.0 -15.0
vsys U PARAS -20.896 0.1 -25.0 -15.0
```

The config file will be read and all parameters are stored in the
```config``` object. Different numbers of instruments and filters
should be handled automatically.
The following output is made:

   1. plot of each variable parameter's walker
   1. corner plot of the post-burnin chain segment
   1. plot of data with fitted model
   1. output of the best fitting parameters

# Contributors

James McCormac, James Blake

