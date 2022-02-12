import argparse
import os

default_parameter_file = '@lcfit.par'


def argparser():
    """
    Creates an argparse.ArgumentParser object for reading in parameters from a file.
    :return:
    """
    ap = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                 description='Train and deploy a deep-learned [Fe/H] estimator'
                                             'based on Gaia time-series photometry.',
                                 epilog="")

    # use custom line parser for the parameter file
    ap.convert_arg_line_to_args = convert_arg_line_to_args

    ap.add_argument('-v',
                    '--verbose',
                    action='store_true',  # assign True value if used
                    help='Generate verbose output.')

    ap.add_argument('--seed',
                    action='store',
                    type=int,
                    default=1,
                    help='Integer seed for reproducible results.')

    ap.add_argument('--rootdir',
                    action='store',
                    type=str,
                    default=os.path.expanduser('~'),
                    help='Full path of the root directory '
                         '(all other directory and file names will be relative to this, default: `~`).')

    ap.add_argument('--input_list',
                    action='store',
                    type=str,
                    default='lcfit_input.lst',
                    help='Name of the input list file (default: `lcfit_input.lst`)')

    ap.add_argument('--input_dir',
                    action='store',
                    type=str,
                    default='.',
                    help='Subdirectory of the input time series (default: `.`)')

    ap.add_argument('--gls_input_file',
                    action='store',
                    type=str,
                    default='gls_input.dat',
                    help='Name of the GLS input file (created by the external code `gls` and read by `lcfit`, '
                         'default: `gls_input.dat`)')

    ap.add_argument('--input_lc_suffix',
                    action='store',
                    type=str,
                    default='.dat',
                    help='Suffix for the input time series file names (default: `.dat`).')

    ap.add_argument('--input_lc_prefix',
                    action='store',
                    type=str,
                    default='',
                    help='Prefix for the input time series file names (default: no prefix).')

    ap.add_argument('--output_dir',
                    action='store',
                    type=str,
                    default='data',
                    help='If specified, the clipped, phased, phase-sorted light curves will be written to '
                         'this directory, into a separate file for each time series (default: `data`).')

    ap.add_argument('--output_syn_dir',
                    action='store',
                    type=str,
                    default=None,
                    help='If specified, the synthetic output time series will be written to '
                         'this directory (separate file for each time series).')

    ap.add_argument('--plot_dir',
                    action='store',
                    type=str,
                    default='plots',
                    help='Subdirectory of the output figures.')

    ap.add_argument('--plot_suffix',
                    action='store',
                    type=str,
                    default='',
                    help='Suffix for the output figures.')

    ap.add_argument('--syn_suffix',
                    action='store',
                    type=str,
                    default='_syn',
                    help='Suffix for the synthetic output time series.')

    ap.add_argument('--output_gpr_dir',
                    action='store',
                    type=str,
                    default='gpr',
                    help='Subdirectory of the output Gaussian Process models.')

    ap.add_argument('--output_param_file',
                    action='store',
                    type=str,
                    default='lcfit.out',
                    help='Name of the output parameter file.')

    ap.add_argument('--output_data_dir',
                    action='store',
                    type=str,
                    default='.',
                    help='Subdirectory for output data files.')

    ap.add_argument('--merged_output_datafile',
                    action='store',
                    type=str,
                    default=None,
                    help='Name of the data file for writing all output time series, appended one after the other.')

    ap.add_argument('--degrade_lc',
                    action='store_true',
                    help='Degrade the light curve (see the source code for the corresponding parameters).')

    ap.add_argument('--compute_errors',
                    action='store_true',  # assign True value if used
                    help='If `gpr_fit` is `True`, then `n_gpr_sample` random samples will be drawn from '
                    'the fitted GPR model at the original (or if the >maxn_gpr, then the '
                    'binned) phases, and each sample will be fitted with a Fourier series. '
                    'The errors of Fourier parameters will be estimated therefrom.')

    ap.add_argument('--pca_model_file',
                    action='store',
                    type=str,
                    default=None,
                    help='Apply the specified PCA transformation to the Fourier parameters.')

    ap.add_argument('--feh_model_file',
                    action='store',
                    type=str,
                    default=None,
                    help='Apply the specified model to predict the [Fe/H] from the Fourier parameters.')

    ap.add_argument('--fold_double_period',
                    action='store_true',  # assign True value if used
                    help='Fold the time series also with twice the period, and create the corresponding output files.')

    ap.add_argument('--loss',
                    action='store',
                    type=str,
                    choices=['linear', 'soft_l1', 'huber', 'cauchy', 'arctan'],
                    default='huber',
                    help='Type of regression loss.')

    ap.add_argument('--sigma_clip_threshold',
                    action='store',
                    type=float,
                    default=None,
                    help='If specified, outliers beyond `sigma_clip_threshold` * residual std. dev. '
                         'will be iteratively omitted.')

    ap.add_argument('--align_phi1',
                    action='store_true',
                    help='The time series will be phase-aligned by the first Fourier phase defined to be at zero.')

    ap.add_argument('--gpr_fit',
                    action='store_true',
                    help='Perform Gaussian Process Regression on the phase-folded time series.')

    ap.add_argument('--lower_length_scale_bound',
                    action='store',
                    type=float,
                    default=0.1,
                    help='Lower length scale bound for the exponential sine-squared kernel. '
                         'Used if the --gpr_fit option is selected. Default: 0.1')

    ap.add_argument('--upper_length_scale_bound',
                    action='store',
                    type=float,
                    default=10.0,
                    help='Upper length scale bound for the exponential sine-squared kernel. '
                         'Used if the --gpr_fit option is selected. Default: 10.0')

    ap.add_argument('--gpr_hparam_optimization',
                    action='store',
                    type=str,
                    choices=['mle', 'cv'],
                    default='mle',
                    help='Method for hyperparameter optimization of the Gaussian Process Regression. '
                         'If `mle`, then the hyperparameters will be optimized by maximizing the log-likelihood '
                         '(this is the widely-used standard method for GPR). '
                         'If `cv`, then the hyperparameters will be optimized by maximizing the R2 cross-validation '
                         'score. The hyperparameter-space will be searched using a Bayesian approach as implemented in '
                         'the `scikit-optimize` API, and 10-fold CV will be used for evaluation.')

    ap.add_argument('--gpr_cv_n_init',
                    action='store',
                    type=int,
                    default=10,
                    help='The number of initial estimates for Bayesian hyperparameter optimization if '
                         '`gpr_hparam_optimization` is `cv`.')

    ap.add_argument('--gpr_cv_n_calls',
                    action='store',
                    type=int,
                    default=10,
                    help='The number of iterations for Bayesian hyperparameter optimization if '
                         '`gpr_hparam_optimization` is `cv`.')

    ap.add_argument('--fourier_from_gpr',
                    action='store_true',
                    help='Infer the Fourier parameters from the best-fitting Gaussian Process Regression model.')

    ap.add_argument('--n_augment_data',
                    action='store',
                    type=int,
                    default=None,
                    help='If specified, the input time series will be augmented by drawing `n_augment_data` new '
                         'time series from the GPR model at the original phase points.')

    ap.add_argument('--plot_augmented',
                    action='store_true',
                    help='Plot the augmented time series.')

    ap.add_argument('--k_fold',
                    action='store',
                    type=int,
                    default=10,
                    help='Number of cross-validation folds.')

    ap.add_argument('--fourier_order_min',
                    action='store',
                    type=int,
                    default=3,
                    help='Minimum Fourier order to consider.')

    ap.add_argument('--fourier_order_max',
                    action='store',
                    type=int,
                    default=8,
                    help='Maximum Fourier order to consider.')

    ap.add_argument('--is_err_col',
                    action='store_true',
                    help='The input contains (a) column(s) with the magnitudes\' uncertainties.')

    ap.add_argument('--is_zperr_col',
                    action='store_true',
                    help='The input contains (a) column(s) with zero-point uncertainties.')

    ap.add_argument('--n_data_cols',
                    action='store',
                    type=int,
                    default=1,
                    help='Number of data columns per input file.')

    ap.add_argument('--use_data_cols',
                    action='store',
                    nargs='*',
                    type=int,
                    default='1',
                    help='List of data columns to use.')

    ap.add_argument('--known_data_cols',
                    action='store_true',
                    help='If set, the input list file is expected to contain an extra column with the '
                         'number of the data column to be used for regression.')

    ap.add_argument('--min_snr',
                    action='store',
                    type=float,
                    default=None,
                    help='If specified, a column with S/N is expected to follow each magnitude column '
                         '(or magnitude error column if included). Data points with values below `min_snr` '
                         'will be rejected.')

    ap.add_argument('--flag_omit',
                    action='store',
                    type=str,
                    default=None,
                    help='If specified, a column with string flags is expected to follow each magnitude column '
                         '(or magnitude error column / S/N column if included). Data points with values `flag_omit` '
                         'will be rejected.')

    ap.add_argument('--mean_phase2',
                    action='store',
                    type=float,
                    default=None,
                    help='Population mean of the phi2 Fourier parameter.')

    ap.add_argument('--mean_phase3',
                    action='store',
                    type=float,
                    default=None,
                    help='Population mean of the phi3 Fourier parameter.')

    ap.add_argument('--mean_phase21',
                    action='store',
                    type=float,
                    default=None,
                    help='Population mean of the phi21 Fourier parameter.')

    ap.add_argument('--mean_phase31',
                    action='store',
                    type=float,
                    default=None,
                    help='Population mean of the phi31 Fourier parameter.')

    ap.add_argument('--waveband',
                    action='store',
                    type=str,
                    default=None,
                    help='Name of the waveband (filter) of the input time series.')

    ap.add_argument('--plot_data',
                    action='store_true',
                    help='Plot the time series.')

    ap.add_argument('--save_figures',
                    action='store_true',
                    help='Save output plots into files.')

    ap.add_argument('--plot_gpr',
                    action='store_true',
                    help='Plot the results of the Gaussian Process Regression.')

    ap.add_argument('--plot_all_datasets',
                    action='store_true',
                    help='Create plots for all datasets.')

    ap.add_argument('--fit_period',
                    action='store_true',
                    help='Fit the period to the dataset (by non-linear least squares regression).')

    ap.add_argument('--do_gls',
                    action='store_true',
                    help='Compute an initial period estimate by the Generalized Lomb-Scargle Periodogram '
                         'method. Otherwise an initial period value is expected in the input file.')

    ap.add_argument('--redo_gls',
                    action='store_true',
                    help='Recompute the initial period estimate by the Generalized Lomb-Scargle Periodogram '
                         'method in the outlier rejection iterations.')

    ap.add_argument('--weighted_fit',
                    action='store_true',
                    help='Use squared inverse of the errors as weights in the regression.')

    ap.add_argument('--nsyn',
                    action='store',
                    type=int,
                    default=120,
                    help='Number of equidistant phase points for the output synthetic data '
                         '(i.e., points of evaluation of the regression model.)')

    ap.add_argument('--min_ndata',
                    action='store',
                    type=int,
                    default=20,
                    help='Minimum number of data points in the time series to be considered for analysis.')

    ap.add_argument('--n_jobs',
                    action='store',
                    type=int,
                    default=None,
                    help='The number parallel threads. If not specified, n_jobs=max(n_cores, k_fold)')

    return ap


def convert_arg_line_to_args(arg_line):
    """
    Custom line parser for argparse.
    :param arg_line: str
    One line of the input parameter file.
    :return: None
    """
    if arg_line:
        if arg_line[0] == '#':
            return
        for arg in arg_line.split():
            if not arg.strip():
                continue
            if '#' in arg:
                break
            yield arg
