# lcfit

Lcfit is a Python3 library for the robust regression of periodic time series.
Its main intended purpose is to serve as a fast and versatile tool for the 
automated batch processing of a very large number of photometric light curves 
of periodic variable stars (e.g., pulsating stars) from large time-domain 
astronomical surveys such as 
[Gaia](https://www.esa.int/Science_Exploration/Space_Science/Gaia_overview), 
[VVV](https://vvvsurvey.org/), 
and [OGLE](https://ogle.astrouw.edu.pl/).
For example, this library was employed in the recent studies by Dekany et al. 
([2019](https://arxiv.org/abs/1908.08290), 
[2020](https://arxiv.org/abs/2006.09883), 
[2021](https://arxiv.org/abs/2107.05983)).

If lcfit is used in an astronomical research project as provided here or in a
modified form, the citation of any of the aforementioned papers in the resulting 
publication will be much appreciated.

## Installation

Simply use `git clone <URL>` to clone this library into a local directory. 
Subsequently, it can be added to your `PYTHONPATH` either temporarily by 
issuing the `sys.path.append("/your/local/directory")` command in python,
or permanantly by exporting you directory into the `PYTHONPATH` system variable.
For example, if using the bash shell, add the 
`export PYTHONPATH="${PYTHONPATH}:/your/local/directory"` in the ~/.bashrc
file.

### Dependencies

Lcfit requires python >3.6 and the 
[numpy](https://numpy.org/), 
[scipy](https://scipy.org/), 
[scikit-learn](https://scikit-learn.org/stable/), 
[scikit-optimize](https://scikit-optimize.github.io/stable/),
[matplotlib](https://matplotlib.org/), 
and [joblib](https://joblib.readthedocs.io/en/latest/) 
python libraries for its full functionality.
In addition, installation of the [gls]() Fortran90 library of this repository 
is required for performing initial period search. The compiled Fortran binary is called by lcfit 
as a subprocess.

The latest version of lcfit was tested under the following python3 environment:
numpy 1.18.5, scipy 1.6.3, scikit-learn 0.24.2, scikit-optimize 0.9.0, 
matplotlib 3.4.2, joblib 1.0.1.

## Usage
Lcfit can be run using command-line arguments:
`python lcfit.py [OPTION]`,
or by supplying a parameter file that includes the command-line arguments:
`python lcfit.py @<parameter_file>`.

The full list of command-line options can be printed on the STDOUT by:
`python lcfit.py --help`.

An example parameter file `lcfit.par` is also provided for convenience.

### Input
Lcfit batch-processes a list of time series stored in separate files, specified 
by the `--input_list <listfile>` parameter. `<listfile>` must contain the following 
columns, separated by whitespaces: 
identifier (string), [initial period _(float)_], [dataset _(int)_]. By default, each 
time-series will be read from the file ./<identifier>.dat. The subdirectory and the
file suffix can be customized by the `--input_dir <dir>` and the 
`--input_lc_suffix <suffix>` parameters. If `--do_gls` is specified, then an
initial value for the period will be computed by the 
[Generalized Lomb-Scargle algorithm](https://arxiv.org/abs/0901.2573) 
(using [this](https://github.com/idekany/gls) implementation).

The time-series files should contain the following columns separated by whitespaces:
time _(float)_, 
value1 _(float)_, [value_err1 _(float)_] [snr1 _(float)_] [flag1 _(str)_], 
[value2 _(float)_, [value_err2 _(float)_] [snr2 _(float)_] [flag2 _(str)_]],
...,
[ZP_error _(float)_]

The numbered columns refer to different alternative datasets (for example, 
photometry of the same star acquired with different aperture sizes). 
By default, a single dataset per input file is expected, this can be changed 
with the `--n_data_cols <number>` parameter. When the input files contain 
multiple datasets, a subset of these to be considered by lcfit can be specified
by the `--use_data_cols <list of numbers>` parameter.

By default, the input files are expected to contain a (time, value) column
pair for each dataset. An error column per dataset can be added by using the 
`--is_err_col` option; by which one can specify the statistical uncertainties
in the measurements of each dataset, and use it for sample weighing in the period
search and/or regression. In addition, a separate column with
the systematic zero-point (ZP) uncertainty (per dataset) can also be added using 
the `--is_zperr_col` option. This can be useful for example if the data have a 
point-to-point calibration of the photometric ZP, such as in the case of 
[VVV](https://vvvsurvey.org/). Finally, columns with the signal-to-noise ratio
of individual measurements and a measurement flag can also be added for each 
dataset by using the `--min_snr <snr>` and `--flag_omit <flag>` options. 
In this case, data with S/N<`<snr>` and/or
flagged with `<flag>` will be omitted from the analysis. 

### Main features

Lcfit performs robust non-linear regression of a truncated Fourier series in
tandem with iterative outlier rejection. The optimal Fourier order (i.e., the
optimal number of Fourier terms) is determined by phase-stratified k-fold 
cross-validation. Subsequently, Gaussian Process Regression (GPR) is performed 
on the resulting phase-folded time series, and the Fourier representation of
the resulting best-fit GPR is provided. Errors in the derived model parameters
can be estimated from the GPR model.

A simplified summary of lcfit in pseudocode:

```angular2html
for each datafile:
    for each dataset:
        initialize Period (P_init) [1]
        repeat:
            optimize Fourier order using P_init via CV [2]
            non-linear regression with optimal Fourier order [3]
            detect outliers around regression model [4]
            if no outliers:
                break
            reassign P_init [5]
    select best dataset (by highest S/N)
    compute phase-folded time-series
    compute GPR model and its Fourier representation [6]
    compute model parameter errors from GPR model [7]
```

[1] The initial value for the period is either supplied by the user in the second 
column of the input list file, or computed by the GLS algorithm (`--do_gls` option).

[2] The number of folds can be specified by the `--k_fold <number>` option. 
The folds are stratified by the phase computed from the latest value of the 
period. The range of possible orders can be specified by the 
`--fourier_order_min <min_order>` and 
`--fourier_order_max <max_order>` parameters.

[3] Regression is done using the trusted-region-reflective algorithm implemented 
in the scipy library. The Huber loss is used by default, and can be changed with 
the `--loss <loss>` parameter. If the `--weighted_fit` option is selected, then 
sample weights proportional to the squared inverse of the uncertainties will 
be used. 

[4] Data points with residual values beyond n * sigma from the regression model 
will be omitted from the dataset, where sigma is the residual standard deviation 
of the latest model, and n can be set by the `--sigma_clip_threshold <n>` 
parameter (default: None, i.e., no outlier rejection).

[5] If `--redo_gls` is set, then the period will be reinitialized at the end of 
every iteration with the GLS algorithm. This can be useful is the dataset contains 
strong outliers that can severely bias the initial estimate of the period.

[6] The GPR is performed on the phase-folded time series using the implementation 
provided by scikit-learn. A scaled periodic kernel (constant kernel * exponential 
sine-squared kernel) is combined with a white kernel, and the noise in the dataset 
is estimated from the latter. Periodic boundary conditions are created by padding 
the folded time series with itself. Hyperparameter optimization is done either by 
the standard method of maximizing the log-marginal likelihood 
(`--gpr_hparam_optimization mle`), or by maximizing the R2-score measured via
10-fold cross-validation (`--gpr_hparam_optimization cv`). For the latter, 
Bayesian optimization is performed for fast convergence, using the scikit-optimize 
implementation. Cross-validation can be advantageous in case of low number of 
noisy datapoints, that can otherwise yield to high-variance GPR models with the 
standard approach. However, maximization of the log-marginal likelihood is 
recommended in general, due to its fast computation.

[7] Errors in the Fourier model parameters are estimated (`--compute_errors`) 
by drawing random samples from the GPR model, and performing regression on 
each realization. The standard deviations of the parameters cross the 
realizations are returned as uncertainty estimates. 

### Output

The derived regression parameters and descriptive statistics are written to
a file (specified by `--output_param_file <filename>`, default: `lcfit.out`,
one line per input file).
The outlier-free, phase-folded time series corresponding to the best dataset 
per input file is written to a subdirecory specified by 
`--output_data_dir <directory_name>`. Synthetic time series (i.e., the 
best-fitting regression model evaluated over an equidistant phase grid) are 
written to the subdirectory specified by `--output_syn_dir <directory_name>`.
If the option `--output_gpr_dir <directory_name>` is provided, the GPR models
will be saved using `joblib` and written to files the provided directory.
Figures showing the phase-folded time series, the rejected outliers, and the 
best-fitting regression models are generated for each input time series and
written to a subdirectory specified by `--plot_dir <directory_name>`.

### Examples

A few example data files are provided in the `test_photometry` subdirectory. 
The input list file `test1.lst` includes 3 RRab Lyrae stars from the OGLE-IV 
survey, their corresponding data files contain a single I-band dataset with 
an error column. 
To analyze these with lcfit, the following options should be set in the 
parameter file `lcfit.par`:
`--input_list test1.lst --input_dir test_photometry --n_data_cols 1 
--use_data_cols 1 --is_err_col`
The input list file `test2.lst` comprises a single RRab Lyrae, and its data file
contains 5 Ks-band datasets from the VVV survey, corresponding to 5 photometric 
apertures of different diameters. The object is temporally blended by a nearby 
point-source, severely contaminating its photometric light curve. Each dataset 
includes an error and a ZP-error column. To analyze the data of this object and 
select the optimal one among the 3 smallest apertures, we should use the following 
settings in `lcfit.par`:
`--input_list test2.lst --input_dir test_photometry --n_data_cols 5 
--use_data_cols 1 2 3 --is_err_col --is_zperr_col`.
The results and output files will be written to the subdirectories provided
with this repository.

## License

[MIT](https://choosealicense.com/licenses/mit/), see `LICENSE`.