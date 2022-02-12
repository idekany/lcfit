import joblib
import lcfit_io as io
import os
import sys
import fourier as ff
import multiprocessing
import numpy as np
import lcfit_utils as ut
from joblib import Parallel, delayed, load
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from time import time
from os.path import isfile

tic = time()

# ======================================================================================================================
#                                              P R E A M B L E

# ------------------------------
#  WIRED-IN PARAMETERS:

phase_ext_neg = 0                       # negative phase range extension beyond [0,1]
phase_ext_pos = 1.2                     # positive phase range extension beyond [0,1]
minplotlcphase = -0.05                  # minimum phase to plot
maxplotlcphase = 2.05                   # maximum phase to plot
constrain_yaxis_range = False           # if True, the y-axis will be constrained to unclipped data
aspect_ratio = 0.6                      # aspect ratio of the plots
figformat = "png"                       # format of the plots
c_freq_tol = 10                         # relative tolerance parameter for the period fit
                                        # (range will be c_freq_tol / total timespan)
gpr_sample_fpars = False                # If `True`, then the sampled Fourier parameters during error estimation
                                        # will be written to a file.
epsilon = 0.05                          # Huber loss L1 -> L2 threshold in the Huber loss function
tol1 = 1e-4                             # tolerance for convergence in the scipy least_squares method
n_gpr_restarts = 0
n_gpr_sample = 100                      # How many samples to draw from the GPR model for error estimation?
maxn_gpr = 200                          # Maximum number of points to be fitted with GPR (or data will be binned)
eps = 0.000001                          # a small positive number :)

# Parameters for data degradation:
remove_points = True
nkeep = 50
add_noise = True
sigma_noise = 0.05
add_phasegap = True
gap_pos = None
gap_length = 0.1
add_outliers = True
sigma_outliers = 0.2
frac_outliers = 0.1

# ------------------------------

# Read parameters from a file or from the command line:
parser = io.argparser()
# print(len(sys.argv))
if len(sys.argv) == 1:
    # use default name for the parameter file
    print(sys.argv)
    pars = parser.parse_args([io.default_parameter_file])
else:
    pars = parser.parse_args()

np.random.seed(pars.seed)

# Define phase grid:
phas = np.linspace(phase_ext_neg, phase_ext_pos, num=pars.nsyn, endpoint=False)

# Load PCA model if required:
if pars.pca_model_file is not None:
    pca_model = load(os.path.join(pars.rootdir, pars.pca_model_file))
else:
    pca_model = None

# Load predictive model for the computation of the [Fe/H] if required:
if pars.feh_model_file is not None:
    try:
        feh_model = load(os.path.join(pars.rootdir, pars.feh_model_file))
        compute_feh = True
    except:
        ut.warn("WARNING: " + pars.feh_model_file + " could not be opened!\n [Fe/H] will not be predicted.")
        compute_feh = False
        feh_model = None
else:
    feh_model = None

# Set the number of parallel threads:
num_cores = multiprocessing.cpu_count()
if pars.n_jobs is not None:
    assert pars.n_jobs >= 1 or pars.n_jobs == -1, "`n_jobs` must be a non-negative integer or -1"
    if pars.n_jobs == -1:
        n_jobs = num_cores
    else:
        n_jobs = pars.n_jobs
else:
    n_jobs = pars.k_fold
if n_jobs > num_cores:
    ut.warn("n_cores has been set to {} (maximum number of cores)".format(num_cores))
    n_jobs = num_cores

if gpr_sample_fpars:
    compute_errors = True

# Define grid of Fourier orders to try
forder_grid = np.arange(pars.fourier_order_min, pars.fourier_order_max + 1, 1)

if not pars.fit_period:
    c_freq_tol = 1e-10

if pars.waveband is not None:
    lcplot_ylabel = '$' + pars.waveband + '$ [mag]'
else:
    lcplot_ylabel = 'mag.'

# END OF PREAMBLE
# ======================================================================================================================


# Read input list:
object_id, object_per, object_dataset = ut.read_input(os.path.join(pars.rootdir, pars.input_list),
                                                      do_gls=pars.do_gls, known_columns=pars.known_data_cols)
n_object = len(object_id)

# ---------------------
# Loop over list of input light-curves:

for iobj, objname in enumerate(object_id):

    objname = str(objname)

    if pars.verbose:
        print("===================> OBJECT {} : {}<==================="
              .format(iobj + 1, objname))
        print("object {} ({})".format(iobj + 1, objname), file=sys.stderr)

    lcfile = os.path.join(pars.rootdir, pars.input_dir, pars.input_lc_prefix + str(objname) + pars.input_lc_suffix)

    # Check if light curve file exists; if not, skip this object and go to the next:
    if not isfile(lcfile):
        ut.warn("Found no data for object {}".format(objname))
        continue

    # Read light curve:
    lcdatain = ut.read_lc(lcfile, n_data_cols=pars.n_data_cols,
                          is_err_col=pars.is_err_col, is_zperr_col=pars.is_zperr_col,
                          flag_column=(pars.flag_omit is not None), snr_column=(pars.min_snr is not None))

    # if lcdatain['otime'].shape[0] < pars.min_ndata:
    if lcdatain.size < pars.min_ndata:
        ut.warn("Object {} : found {} data point(s) on input ({} required), object was skipped."
                .format(objname, lcdatain.size, pars.min_ndata))
        continue

    # Initialize variables:
    snr_best = -1  # initialize cost for dff fit
    best_dataset = None  # initialize value for optimal dataset
    fm_best = None
    period_o_best = None
    arrays_best = None

    if pars.known_data_cols:
        pars.use_data_cols = [object_dataset[iobj]]
        print("dataset: " + str(pars.use_data_cols))

    # Truncate time values:
    otime0 = lcdatain['otime'][0]

    # Loop over datasets:
    for idata in np.array(pars.use_data_cols) - 1:

        if pars.verbose:
            print("\n----------\nDataset {}\n----------".format(idata + 1))

        lcdatain = lcdatain[~np.isnan(lcdatain['mag' + str(idata + 1)])]

        if pars.flag_omit is not None:
            lcdatain = lcdatain[lcdatain['flag' + str(idata + 1)].astype(str) != pars.flag_omit]
        if pars.min_snr is not None:
            lcdatain = lcdatain[lcdatain['snr' + str(idata + 1)] >= pars.min_snr]

        mag = lcdatain['mag' + str(idata + 1)]
        ndata = len(mag)

        if ndata < pars.min_ndata:
            ut.warn("Object {} : found only {} data points within constraints in dataset {} ({} required), "
                    "dataset was skipped."
                    .format(objname, ndata, idata + 1, pars.min_ndata, idata + 1))
            continue

        otime = lcdatain['otime'] - otime0

        if pars.is_zperr_col:
            zperr = lcdatain['zperr' + str(idata + 1)]
        else:
            zperr = np.zeros(mag.shape)

        if pars.is_err_col:
            magerr = lcdatain['magerr' + str(idata + 1)]
            merr = np.sqrt(magerr ** 2 + zperr ** 2)
        else:
            magerr = np.zeros_like(mag)
            merr = np.sqrt((magerr + eps) ** 2 + zperr ** 2)

        if pars.verbose:
            print("n_LC = " + str(ndata))

        # Determine initial period:
        if pars.do_gls:
            period = ff.glsper(otime, mag, merr, glsinputfile=pars.gls_input_file)
            if pars.verbose:
                print("P_GLS = {}".format(period))
        else:
            period = object_per[iobj]
            if pars.verbose:
                print("P_in = {}".format(period))

#       ------------------------------------------
        # degrade light curve

        if pars.degrade_lc:
            otime, mag, magerr, zperr, rem_inds = \
                ut.degrade_lc(otime, mag, magerr, zperr, period=period, remove_points=remove_points, nkeep=nkeep,
                              min_otime=None, max_otime=None,
                              add_noise=add_noise, sigma_noise=sigma_noise,
                              add_phasegap=add_phasegap, gap_pos=gap_pos, gap_length=gap_length,
                              add_outliers=add_outliers, sigma_outliers=sigma_outliers, frac_outliers=frac_outliers,
                              verbose=pars.verbose)

            print("Degraded light curve:")
            ndata = len(mag)
            print("n_LC = " + str(ndata))
            merr = np.sqrt(magerr ** 2 + zperr ** 2)

#       ------------------------------------------

        if pars.weighted_fit and pars.is_err_col:
            weights = ((merr + eps) * 100) ** (-2)
        else:
            weights = np.ones(magerr.shape[0])

        # Save original data before further steps:
        otime_o, mag_o, magerr_o, zperr_o = np.copy(otime), np.copy(mag), np.copy(magerr), np.copy(zperr)
        period_o = period
        ndata_o = ndata

        #  Non-linear regression of truncated Fourier series with Huber loss function and iterative outlier rejection,
        #   the Fourier order is optimized via k-fold cross-validation.
        if pars.verbose:
            print("---------- Direct Fourier fit ----------\n")

        fm = ff.FourierModel(c_freq_tol=1e-10, loss=pars.loss, epsilon=epsilon,
                             mean_phase2=pars.mean_phase2, mean_phase3=pars.mean_phase3,
                             mean_phase21=pars.mean_phase21, mean_phase31=pars.mean_phase31,
                             tol=tol1, verbose=False)

        io = 0

        while True:
            io = io + 1
            if pars.verbose:
                print("Iteration {}".format(io))
            ndata = len(mag)
            if ndata < pars.k_fold:
                pars.k_fold = ndata

            phase_obs = ff.get_phases(period, otime, epoch=0.0, shift=0.0)  # compute phases with input period
            splitter = StratifiedKFold(n_splits=pars.k_fold, shuffle=True, random_state=pars.seed)
            cv_folds = list(splitter.split(mag, ut.get_stratification_labels(phase_obs, pars.k_fold)))
            val_scores = []

            fm.set_params({'period_': period})

            for forder in forder_grid:  # Start loop over grid of Fourier orders

                fm.set_params({'order': forder, 'c_freq_tol': 1e-10})

                cv_output = np.array(Parallel(n_jobs=n_jobs)(
                    delayed(ut.fit_validate_model)(fm, otime, mag, train_index, val_index, weights=weights) for
                    train_index, val_index in cv_folds), dtype=object)

                mag_val = np.concatenate(cv_output[:, 0]).astype(float)
                pred_val = np.concatenate(cv_output[:, 1]).astype(float)
                weights_val = np.concatenate(cv_output[:, 2]).astype(float)
                val_score = mean_squared_error(mag_val, pred_val, sample_weight=weights_val, squared=False)
                val_scores.append(val_score)

                if pars.verbose:
                    print("order = {0}  --->  score = {1:.8f}".format(forder, val_score))

            val_scores = np.array(val_scores)
            max_score_ind = np.unravel_index(val_scores.argmin(), val_scores.shape)[0]
            max_score = val_scores[max_score_ind]
            forder_opt = forder_grid[max_score_ind]

            if pars.verbose:
                print("\noptimal order = {0}  (mean CV score = {1:.3f})".format(forder_opt, max_score))

            # Now that we know the best Fourier order, we allow period to be fitted (if pars.fit_period is True),
            # the value of c_freq_tol is used to define the trusted region.
            if pars.fit_period:
                fm.set_params({'c_freq_tol': c_freq_tol})
                # fm.fit(otime, mag, weights=weights, predict=False)
            fm.set_params({'order': forder_opt})
            # fm.set_params({'c_freq_tol': 1e-10, 'order': forder_opt})
            prediction, residual = fm.fit(otime, mag, weights=weights, predict=True)
            fm.compute_results(phas)

            rstdev = np.std(residual)  # std. dev. of the residual

            # Search for outliers:
            if pars.sigma_clip_threshold is None:
                break
            keepmask = (abs(residual) <= pars.sigma_clip_threshold * rstdev)
            nomit = np.sum(np.invert(keepmask))  # number of data points to be omitted
            if pars.verbose:
                print("number of outliers: {}".format(nomit))
            if nomit == 0:
                break
            # Omit outliers:
            mag, otime, magerr, zperr, weights, merr = \
                mag[keepmask], otime[keepmask], magerr[keepmask], zperr[keepmask], weights[keepmask], merr[keepmask]

            if pars.fit_period and pars.redo_gls:
                period = ff.glsper(otime, mag, merr, glsinputfile=pars.gls_input_file)
                if pars.verbose:
                    print("P_GLS = {}".format(period))
            else:
                period = fm.period_
            # print("fitted period", period)

        if pars.verbose:
            print("\nobject: {0}  ,   N={1} ({2})  ,  ap. {3}:  N_F={4:d}  ,  P={5:.6f}  ,  dP={6:.6f}  ,  "
                  "sig={7:.3f}  ,  cost={8:.4f}  ,  <mag>={9:.3f}  ,  SNR={10:.2f}"
                  .format(objname, ndata, ndata_o, idata + 1, forder_opt, fm.period_,
                          fm.period_ - period_o, rstdev, fm.cost_, fm.intercept_, fm.results_['snr']))

        if fm.results_['snr'] > snr_best:  # check if the solution is better than the one for the previous dataset

            fm_best = fm
            snr_best = fm.results_['snr']
            best_dataset = idata
            period_o_best = period_o
            arrays_best = (otime, otime_o, mag, mag_o, magerr, magerr_o, zperr, zperr_o)

        if pars.plot_all_datasets and len(pars.use_data_cols) > 1:

            # Create a plot for each dataset of this object.
            shift = fm.phases_[0] / (2 * np.pi)
            phase_obs = ff.get_phases(fm.period_, otime, epoch=0.0, shift=shift, all_positive=True)
            phase_obs_o = ff.get_phases(fm.period_, otime_o, epoch=0.0, shift=shift, all_positive=True)
            syn = fm.predict(phas, shift=shift, for_phases=True)
            outfile = os.path.join(pars.rootdir, pars.plot_dir, objname + "__" + str(idata + 1) + ".png")
            ut.plotlc(
                (np.vstack((phase_obs_o, mag_o, magerr_o)).T, np.vstack((phase_obs, mag, magerr)).T,
                 np.vstack((phas, syn)).T),
                symbols=("ro", "ko", "r-"), title=objname,
                figtext="dataset: {0} , $P_fit$={1:.6f} , $N_F$={2}".format(idata + 1, fm.period_, forder_opt),
                figsave=pars.save_figures, outfile=outfile)

    # ====> END OF LOOP OVER DATASETS

    if best_dataset is None:  # In this case, all columns were rejected (not enough data within constrains).
        ut.warn("Not enough data within constraints, all columns were rejected, object {} is skipped."
                .format(objname))
        continue

    (otime, otime_o, mag, mag_o, magerr, magerr_o, zperr, zperr_o) = arrays_best

    # Do we want to shift the phases so that the zero phase is at the first Fourier phase?
    if pars.align_phi1:
        shift = fm_best.phases_[0] / (2 * np.pi)
    else:
        shift = 0.0

    # Phase-fold the light curve, adjust phases:
    phase_obs = ff.get_phases(fm_best.period_, otime, epoch=0.0, shift=shift, all_positive=True)
    phase_obs_o = ff.get_phases(fm_best.period_, otime_o, epoch=0.0, shift=shift, all_positive=True)
    phasecov1 = ff.phase_coverage(phase_obs)

    # Phase-fold the light curve with DOUBLE PERIOD, adjust phases :
    phase_obs_2p = ff.get_phases(fm_best.period_ * 2, otime, epoch=0.0, shift=shift / 2.0, all_positive=True)
    phase_obs_o_2p = ff.get_phases(fm_best.period_ * 2, otime_o, epoch=0.0, shift=shift / 2.0, all_positive=True)
    phasecov2 = ff.phase_coverage(phase_obs_2p)

    nepoch = fm_best.ndata
    minmax = np.max(mag) - np.min(mag)

    totalzperr = np.sqrt(np.sum(zperr ** 2)) / fm_best.ndata

    # Save results in dictionary:
    # fm_best.compute_results(phas)
    results = fm_best.results_
    results.update(
        {'objname': objname, 'delta_per': fm_best.period_ - period_o_best, 'nepoch': nepoch, 'minmax': minmax,
         'otime': otime, 'otime_o': otime_o, 'otime0': otime0,
         'mag': mag, 'mag_o': mag_o, 'magerr': magerr, 'magerr_o': magerr_o, 'dataset': best_dataset,
         'ph': phase_obs, 'ph_2p': phase_obs_2p, 'ph_o': phase_obs_o, 'ph_o_2p': phase_obs_o_2p,
         'zperr_o': zperr_o, 'zperr': zperr, 'phcov': phasecov1, 'phcov2': phasecov2,
         'totalzperr': totalzperr, 'period': fm_best.period_, 'cost': fm_best.cost_, 'forder': fm_best.order})
    if feh_model is not None:
        # feh = ut.smolec_feh(fm_best.period_, fm_best.phi31_, fm_best.amplitudes_[1])
        feh = feh_model.predict(np.array([[fm_best.period_, fm_best.amplitudes_[1], fm_best.phi31_]]))
        results.update({'feh': feh})
    if pars.pca_model_file is not None:
        pca_feat = ff.pca_transform(pca_model, results)
        results.update({'pca_feat': pca_feat})

    if pars.verbose:
        print("============================================================\nRESULTS:")

    print("object: {0}  ,   N={1} ({2})  ,  ap. {3}:  N_F={4:d}  ,  P={5:.6f}  ,  dP={6:.6f}  ,  sig={7:.3f}  ,"
          "  cost={8:.4f}  ,  <mag>={9:.3f},  SNR={10:.2f}"
          .format(objname, results['ndata'], ndata_o, best_dataset + 1, results['forder'], results['period'],
                  results['delta_per'], results['stdv'], results['cost'], results['icept'], results['snr']))
    if pars.verbose:
        print("Intercept: {}".format(results['icept']))
        print("Amplitudes: {}".format(results['A']))
        print("phi21 = {}".format(results['phi21']))
        print("phi31 = {}".format(results['phi31']))

    # Perform GPR fit on the phase-folded light curve:
    if pars.gpr_fit:

        gprm = ff.GPRModel(maxn_gpr=maxn_gpr, phase_ext_neg=phase_ext_neg, phase_ext_pos=phase_ext_pos,
                           n_restarts_optimizer=n_gpr_restarts, hparam_optimization=pars.gpr_hparam_optimization,
                           n_init=pars.gpr_cv_n_init, n_calls=pars.gpr_cv_n_calls,
                           lower_length_scale_bound=pars.lower_length_scale_bound,
                           upper_length_scale_bound=pars.upper_length_scale_bound, n_jobs=n_jobs)

        gprm.fit(results['ph'], results['mag'], noise_level=results['stdv'],
                 verbose=pars.verbose, random_state=pars.seed)

        synmag_gpr, sigma_gpr = gprm.predict(phas.reshape(-1, 1), return_std=True)
        results.update({'synmag_gpr': synmag_gpr, 'sigma_gpr': sigma_gpr})

        if pars.fourier_from_gpr:
            # Fit a fix 20-order Fourier-series to the GPR model evaluated over the phases in 'phas'
            print("Computing Fourier parameters from GPR model...")
            fm_best.set_params({'c_freq_tol': 1e-10, 'period_': 1.0, 'order': 20})
            fm_best.fit(phas, synmag_gpr, weights=None, predict=False)
            fm_best.compute_results(phas, data=(results['ph'], results['mag']), shiftphase=pars.align_phi1)
            results.update(fm_best.results_)
            if pars.pca_model_file is not None:
                pca_feat = ff.pca_transform(pca_model, results)
                results.update({'pca_feat': pca_feat})

        if pars.n_augment_data is not None:
            aug_samples_gpr, synmag_gpa = gprm.augment_data(phas, n_aug=pars.n_augment_data, verbose=True,
                                                            random_state=pars.seed)
            results.update({'synmag_gpa': synmag_gpa})

            # Plot augmented dataset
            if pars.plot_augmented:
                for iaug in range(min(10, pars.n_augment_data)):
                    outfile = os.path.join(pars.rootdir, pars.plot_dir,
                                           objname + "_aug_" + str(iaug + 1) + ".png")
                    figtext = '$P = {0:.6f}$ , augmented #{1}'.format(results['period'], iaug)
                    data1 = np.vstack((results['ph'], results['mag'])).T
                    data2 = np.vstack((gprm._gpr_input_phase, aug_samples_gpr[:, iaug])).T
                    data3 = np.vstack((phas, synmag_gpa[:, iaug])).T
                    data4 = np.vstack((phas, synmag_gpr, sigma_gpr)).T
                    plot_input = (data1, data2, data3, data4)
                    fillerr_index = (3,)
                    symbols = ('kX', 'r.', 'r-', 'b-')
                    ut.plotlc(plot_input, symbols=symbols, fillerr_index=fillerr_index, figsave=pars.save_figures,
                              outfile=outfile, xlabel='phase', ylabel=lcplot_ylabel, figtext=figtext,
                              title=objname)

        if pars.output_gpr_dir is not None:
            joblib.dump(gprm, os.path.join(pars.rootdir, pars.output_gpr_dir, objname + "_gprm.save"))

        if pars.compute_errors:
            print("Computing errors from GPR model...")
            results_err = \
                gprm.get_fourier_errors(fourier_from_gpr=False,
                                        n_samples=n_gpr_sample, order=results['forder'], random_state=pars.seed,
                                        mean_phase2=pars.mean_phase2, mean_phase3=pars.mean_phase3,
                                        mean_phase21=pars.mean_phase21, mean_phase31=pars.mean_phase31,
                                        feh_model=feh_model, period=results['period'],
                                        pca_model=pca_model)

            results.update(results_err)

    # ------------------------------------------------------------------------------------------------------------------
    # C R E A T E    O U T P U T

    # Write results and output data to files, and create plots:

    # Write clipped light curves into a single file, appending the data for each object after one another.
    if pars.merged_output_datafile is not None:
        ut.write_merged_datafile(pars, results)

    # Write out phased, phase-sorted light curve in the phase range [phase_ext_neg, phase_ext_pos]
    # into a separate file for each object.
    if pars.output_data_dir is not None:
        ut.write_single_datafile(pars, results, phase_ext_neg=phase_ext_neg, phase_ext_pos=phase_ext_pos)

    # Write out the fitted model parameters and statistics:
    ut.write_results(pars, results)

    # Write out syntheic time series:
    if pars.output_syn_dir is not None:
        ut.write_synthetic_data(pars, results)

    # Create figures:
    if pars.plot_data:
        ut.make_figures(pars, results, constrain_yaxis_range=constrain_yaxis_range,
                        minphase=minplotlcphase, maxphase=maxplotlcphase, aspect_ratio=aspect_ratio,
                        figformat=figformat)


toc = time()
print("\n--- Execution time: {0:.1f} seconds ---".format(toc - tic))
