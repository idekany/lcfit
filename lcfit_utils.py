# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import fourier as ff
import matplotlib
import warnings
from matplotlib import pyplot as plt
from os.path import isfile

matplotlib.use('Agg')


def warn(*args, **kwargs):
    print('WARNING: ', *args, file=sys.stderr, **kwargs)


def fit_validate_model(model, x: np.array, y: np.array, train_index, val_index, weights: np.array = None):
    x_t, x_v = x[train_index], x[val_index]
    y_t, y_v = y[train_index], y[val_index]
    if weights is not None:
        weights_t, weights_v = weights[train_index], weights[val_index]
    else:
        weights_t = None
        weights_v = None

    # print("y_train:")
    # print(y_t)

    model.fit(x_t, y_t, weights=weights_t)

    yhat_v = model.predict(x_v)

    return y_v, yhat_v, weights_v


def get_stratification_labels(data, n_folds):
    """
    Create an array of stratification labels from an array of continuous values to be used in a stratified cross-
    validation splitter.
    :param data: list or numpy.ndarray
        The input data array.
    :param n_folds: int
        The number of cross-validation folds to be used with the output labels.
    :return: labels, numpy.ndarray
        The array of integer stratification labels.
    """

    assert isinstance(data, np.ndarray or list), "data must be of type list or numpy.ndarray"
    if isinstance(data, list):
        data = np.array(data)

    ndata = len(data)
    isort = np.argsort(data)  # Indices of sorted phases
    labels = np.empty(ndata)
    labels[isort] = np.arange(ndata)  # Compute phase order
    labels = np.floor(labels / n_folds)  # compute phase labels for StratifiedKFold
    if np.min(np.bincount(labels.astype(int))) < n_folds:  # If too few elements are with last label, ...
        labels[labels == np.max(labels)] = np.max(
            labels) - 1  # ... the then change that label to the one preceding it

    return labels


def write_results(pars, results: dict):
    # check if the file already exists:
    newfile = not isfile(os.path.join(pars.rootdir, pars.output_param_file))

    with open(os.path.join(pars.rootdir, pars.output_param_file), 'a') as file:
        if newfile:
            # Write header:
            if pars.compute_errors:
                file.write('# id  Nep  period  totamp  A1  A2  A3  A1_e  A2_e  A3_e  phi1  phi2  phi3  '
                           'phi1_e  phi2_e  phi3_e  phi21  phi21_e  phi31  phi31_e  '
                           'meanmag  meanmag_e  cost  aper  phcov  phcov2  snr  ZPErr  Npt  order  minmax')
            else:
                file.write('# id  Nep  period  totamp  A1  A2  A3  phi1  phi2  phi3  phi21  phi31  meanmag  cost  '
                           'aper  phcov  phcov2  snr  ZPErr  Npt  order  minmax')

            if pars.feh_model_file is not None:
                file.write('  FeH')
                if pars.compute_errors:
                    file.write('  FeH_e')

            if pars.pca_model_file is not None:
                file.write('  E1  E2  E3  E4  E5  E6')
                if pars.compute_errors:
                    file.write('  E1_e  E2_e  E3_e  E4_e  E5_e  E6_e')
            file.write('\n')

        # ------------------------

        if pars.compute_errors:
            file.write(
                "%s %4d %.6f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.4f %.4f %.3f %.3f "
                "%.3f %.3f %.4f %d %.3f %.3f %.1f %.4f %4d %2d %.3f" %
                (results['objname'], results['nepoch'], results['period'], results['tamp'],
                 results['A'][0], results['A'][1], results['A'][2],
                 results['A_std'][0], results['A_std'][1], results['A_std'][2],
                 results['Pha'][0], results['Pha'][1], results['Pha'][2],
                 results['Pha_std'][0], results['Pha_std'][1], results['Pha_std'][2],
                 results['phi21'], results['phi21_std'], results['phi31'], results['phi31_std'],
                 results['icept'], results['icept_std'], results['cost'], results['dataset'] + 1,
                 results['phcov'], results['phcov2'], results['snr'], results['totalzperr'],
                 results['ndata'], results['forder'], results['minmax']))
        else:
            file.write("%s %4d %.6f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.4f %.4f %.3f "
                       "%.4f %d %.3f %.3f %.1f %.4f %4d %2d %.3f" %
                       (results['objname'], results['nepoch'], results['period'], results['tamp'],
                        results['A'][0], results['A'][1], results['A'][2],
                        results['Pha'][0], results['Pha'][1], results['Pha'][2],
                        results['phi21'], results['phi31'],
                        results['icept'], results['cost'], results['dataset'] + 1,
                        results['phcov'], results['phcov2'], results['snr'], results['totalzperr'],
                        results['ndata'], results['forder'], results['minmax']))

        if pars.feh_model_file is not None:
            file.write("  %.3f" % results['feh'])
            if pars.compute_errors:
                file.write("  %.3f" % results['feh_std'])

        if pars.pca_model_file is not None:
            file.write("  %.6f %.6f %.6f %.6f %.6f %.6f" %
                       (results['pca_feat'][0], results['pca_feat'][1], results['pca_feat'][2],
                        results['pca_feat'][3], results['pca_feat'][4], results['pca_feat'][5]))
            if pars.compute_errors:
                file.write("  %.6f %.6f %.6f %.6f %.6f %.6f" %
                           (results['pca_feat_std'][0], results['pca_feat_std'][1], results['pca_feat_std'][2],
                            results['pca_feat_std'][3], results['pca_feat_std'][4], results['pca_feat_std'][5]))

        file.write("\n")


def write_merged_datafile(pars, results: dict):
    # check if the file already exists:
    newfile = not isfile(os.path.join(pars.rootdir, pars.merged_output_datafile))

    with open(os.path.join(pars.rootdir, pars.merged_output_datafile), 'a') as file:
        if newfile:
            file.write('# id  time  mag  mag_err  ZP_err\n')

        outarr = np.rec.fromarrays((np.tile(results['objname'], results['ndata']),
                                    results['otime'] + results['otime0'],
                                    results['mag'], results['magerr'], results['zperr']))
        np.savetxt(file, outarr, fmt='%s %.6f %.3f %.3f %.3f')


def write_single_datafile(pars, results: dict, phase_ext_neg=0, phase_ext_pos=1.2):
    ophase_sorted, mag_sorted = extend_phases(results['ph'], results['mag'],
                                              phase_ext_neg=phase_ext_neg, phase_ext_pos=phase_ext_pos, sort=True)
    outarr = np.rec.fromarrays((ophase_sorted, mag_sorted), names=('phase', 'kmag'))
    with open(os.path.join(pars.rootdir, pars.output_data_dir, results['objname'] + '.dat'), 'w') as file:
        np.savetxt(file, outarr, fmt='%f %f')

    if pars.fold_double_period:
        ophase_sorted2, mag_sorted2 = extend_phases(results['ph_2p'], results['mag'],
                                                    phase_ext_neg=phase_ext_neg, phase_ext_pos=phase_ext_pos, sort=True)
        outarr = np.rec.fromarrays((ophase_sorted2, mag_sorted2), names=('phase', 'kmag'))
        with open(os.path.join(pars.rootdir, pars.output_data_dir, results['objname'] + '_2p.dat'), 'w') as file:
            np.savetxt(file, outarr, fmt='%f %f')


def write_synthetic_data(pars, results: dict):

    if pars.gpr_fit:
        outarr = np.rec.fromarrays((results['phase_grid'], results['synmag_gpr'] - results['icept']))
        np.savetxt(os.path.join(pars.rootdir, pars.output_syn_dir,
                                results['objname'] + "_gpr" + pars.syn_suffix + '.dat'),
                   outarr, fmt='%.4f %.4f')
        if pars.n_augment_data is not None:
            outarr = np.hstack((results['phase_grid'].reshape(-1, 1), (results['synmag_gpr']).reshape(-1, 1), results['synmag_gpa']))
            np.savetxt(os.path.join(pars.rootdir, pars.output_syn_dir,
                                    results['objname'] + "_gpr_aug" + pars.syn_suffix + '.dat'),
                       outarr, fmt='%7.4f ' * (pars.n_augment_data + 2))
    else:
        outarr = np.rec.fromarrays((results['phase_grid'], results['syn'] - results['icept']))
        np.savetxt(os.path.join(pars.rootdir, pars.output_syn_dir,
                                results['objname'] + "_dff" + pars.syn_suffix + '.dat'),
                   outarr, fmt='%.4f %.4f')


def make_figures(pars, results: dict, constrain_yaxis_range=True,
                 minphase=0, maxphase=1.2, aspect_ratio=0.6, figformat: str = 'png'):

    # Create phase diagram:
    outfile = os.path.join(pars.rootdir, pars.plot_dir, results['objname'] + pars.plot_suffix + "." + figformat)

    plottitle = results['objname']
    # plottitle = None
    # figtext = '$P = {0:.6f}$ , $N_F = {1}$ , ap = {2}'.format(results['period'],results['forder'],bestap+1)
    # figtext = '$P = {0:.6f}$'.format(results['period'])
    figtext = '$P = {0:.6f}$ , $S/N = {1:d}$'.format(results['period'], int(results['snr']))

    data1 = np.vstack((results['ph_o'], results['mag_o'], results['magerr_o'])).T
    data2 = np.vstack((results['ph'], results['mag'], results['magerr'])).T
    if pars.fourier_from_gpr:
        data3 = np.vstack((results['phase_grid'], results['synmag_gpr'])).T
    else:
        data3 = np.vstack((results['phase_grid'], results['syn'])).T

    # labels = ("orig.", "clipped", "binned", "DFF")

    if pars.gpr_fit and pars.plot_gpr:
        data4 = np.vstack((results['phase_grid'], results['synmag_gpr'], results['sigma_gpr'])).T
        plot_input = (data1, data2, data3, data4)
        fillerr_index = (3,)
        symbols = ('r.', 'b.', 'r-', 'b-')
    else:
        plot_input = (data1, data2, data3)
        fillerr_index = ()
        symbols = ('r.', 'k.' 'r-')
    plotlc(plot_input, symbols=symbols, fillerr_index=fillerr_index, figsave=pars.save_figures, outfile=outfile,
           xlabel='phase', ylabel='$' + pars.waveband + '$ [mag.]', figtext=figtext, title=plottitle,
           constrain_yaxis_range=constrain_yaxis_range, minphase=minphase, maxphase=maxphase,
           aspect_ratio=aspect_ratio, figformat=figformat)

    if pars.fold_double_period:
        # Create phase diagram with double period:
        outfile = os.path.join(pars.rootdir, pars.plot_dir, results['objname'] + pars.plot_suffix + "_2p." + figformat)
        figtext = '$2P = {0:.6f}$'.format(results['period'] * 2, results['forder'], results['dataset'] + 1)
        data1 = np.vstack(
            (results['ph_o_2p'], results['mag_o'], np.sqrt(results['magerr_o'] ** 2 + results['zperr_o'] ** 2))).T
        data2 = np.vstack(
            (results['ph_2p'], results['mag'], np.sqrt(results['magerr'] ** 2 + results['zperr'] ** 2))).T

        labels = ("orig.", "clipped")
        plot_input = (data1, data2)
        symbols = ('ro', 'ko')

        plotlc(plot_input, symbols=symbols, fillerr_index=(), figsave=pars.save_figures, outfile=outfile,
               xlabel='phase', ylabel='$' + pars.waveband + '$ [mag.]', figtext=figtext, title=results['objname'],
               constrain_yaxis_range=True, figformat=figformat)


def read_input(fname: str, do_gls=False, known_columns=False):
    """
    Reads the input list file with columns: object ID, [period, [dataset]]
    :param fname: string, the name of the input file
    :param do_gls: boolean, whether to perform GLS on the input time series. If False, the second column of the input
    file must contain the period.
    :param known_columns: boolean; whether the dataset to be used is known. If True, the last column of the input
    file must contain the number of the column.
    :return: ndarray(s) or None(s); 1-d arrays with the obect IDs, periods, and datasets
    """
    dtypes = ['|S25']  # dtype for first column: identifiers

    if do_gls:
        if known_columns:
            usecols = (0, 1)
            dtypes = dtypes + ['i']
        else:
            usecols = (0,)
    else:
        if known_columns:
            usecols = (0, 1, 2)
            dtypes = dtypes + ['f8'] + ['i']
        else:
            usecols = (0, 1)
            dtypes = dtypes + ['f8']

    arr = np.genfromtxt(fname, usecols=usecols,
                        dtype=dtypes, unpack=False, comments='#', filling_values=np.nan, names=True)

    object_id = arr['id'].reshape(-1, ).astype(str)

    if do_gls:
        object_per = None
    else:
        object_per = arr['period'].reshape(-1, )

    if known_columns:
        object_ap = arr['ap'].reshape(-1, )
    else:
        object_ap = None

    return object_id, object_per, object_ap


def read_lc(lcfile, n_data_cols: int = 1, is_err_col: bool = False, flag_column: bool = False,
            snr_column: bool = False, is_zperr_col: bool = False, missing_values="NaN", invalid_raise=False):

    assert n_data_cols > 0, "`n_datasets` must be non-zero integer"
    colnames = ['otime']
    dtypes = [float]
    ncols = 1

    for ii in range(n_data_cols):

        colnames.append('mag' + str(ii+1))
        dtypes.append(float)
        ncols += 1

        if is_err_col:
            # We expect the column following each magnitude column to contain the magnitude uncertainty
            colnames.append('magerr' + str(ii + 1))
            dtypes.append(float)
            ncols += 1

        if is_zperr_col:
            # The last column is expected to contain the zero-point error:
            colnames.append('zperr' + str(ii + 1))
            dtypes.append(float)
            ncols += 1

        if snr_column:
            # We expect the next column to contain the S/N
            colnames.append('snr' + str(ii + 1))
            dtypes.append(float)
            ncols += 1

        if flag_column:
            # We expect the next column to contain the flag
            colnames.append('flag' + str(ii + 1))
            dtypes.append('|S10')
            ncols += 1

    used_cols = list(range(ncols))

    # Read light curve:
    lcdatain = np.genfromtxt(lcfile, unpack=False, comments='#', filling_values=np.nan,
                             dtype=dtypes, usecols=used_cols, missing_values=missing_values,
                             names=colnames, invalid_raise=invalid_raise)
    print(lcfile + " found.")

    lcdatain = lcdatain[~np.isnan(lcdatain['otime'])]

    return lcdatain


def degrade_lc(otime, mag, magerr, zperr, period=1.0, remove_points=True, nkeep=50,
               min_otime=None, max_otime=None,
               add_noise=False, sigma_noise=0.05,
               add_phasegap=False, gap_pos=None, gap_length=0.1,
               add_outliers=False, sigma_outliers=0.1, frac_outliers=0.1,
               verbose=False):
    if min_otime is not None:
        mask = (otime > min_otime)
        otime, mag, magerr, zperr = otime[mask], mag[mask], magerr[mask], zperr[mask]

    if max_otime is not None:
        mask = (otime < max_otime)
        otime, mag, magerr, zperr = otime[mask], mag[mask], magerr[mask], zperr[mask]

    if add_phasegap:
        if gap_pos is None:
            # Make the phasegap's position random betwen 0 and 1:
            gap_pos = np.random.random()
        pha = ff.get_phases(period, otime, epoch=0.0, shift=0.0, all_positive=True)
        if gap_pos + gap_length > 1:
            not_gap_inds = [(pha < gap_pos) & (pha > (gap_pos - 1 + gap_length))]
        else:
            not_gap_inds = [(pha < gap_pos) | (pha > (gap_pos + gap_length))]
        mag = mag[not_gap_inds]
        otime = otime[not_gap_inds]
        magerr = magerr[not_gap_inds]
        zperr = zperr[not_gap_inds]

        if verbose:
            print("N_data = {} (after phase gap added)".format(len(mag)))

    if remove_points:
        nremove = len(mag) - nkeep
        if nremove > 0:
            rem_inds = np.random.choice(range(len(mag)), size=nremove, replace=False)
            otime = np.delete(otime, rem_inds)
            mag = np.delete(mag, rem_inds)
            magerr = np.delete(magerr, rem_inds)
            zperr = np.delete(zperr, rem_inds)
            if verbose:
                print("N_data = {} (after points removed)".format(len(mag)))

    out_inds = np.array([])

    if add_outliers:
        out_inds = np.random.choice(range(len(mag)), size=int(len(mag) * frac_outliers), replace=False)
        mag[out_inds] = np.random.normal(mag[out_inds], sigma_outliers)
        if verbose:
            print("{} %% of points made outliers with sigma = {}".format(frac_outliers * 100.0, sigma_outliers))

    if add_noise:
        mag = mag + np.random.normal(mag, sigma_outliers)
        magerr = magerr + sigma_noise

    return otime, mag, magerr, zperr, out_inds


def plotlc(datasets, symbols=(), labels=(), fillerr_index=(), title=None, figtext="",
           minphase=-0.05, maxphase=2.05, figsave=False, outfile=None, invert_y_axis=True,
           constrain_yaxis_range=False, xlabel='phase', ylabel='magnitude', aspect_ratio=0.6, figformat="png"):
    capsize = 1  # size of the error cap

    assert type(datasets) is tuple, "Error: expected tuple for argument, got {}".format(type(datasets))
    assert type(symbols) is tuple, "Error: expected tuple for argument, got {}".format(type(symbols))
    assert (type(labels) is tuple), "Error: expected tuple for argument, got {}".format(type(labels))
    assert (type(figtext) is str), "Error: expected string for argument, got {}".format(type(figtext))

    # Check if there is a title, if yes, adjust plot to make it fit and write it.
    fig = plt.figure(figsize=(6, 6 * aspect_ratio))
    if title is not None:
        if len(labels) > 0:
            fig.subplots_adjust(bottom=0.15, top=0.80, hspace=0.3, left=0.12, right=0.98, wspace=0)
        else:
            fig.subplots_adjust(bottom=0.15, top=0.88, hspace=0.3, left=0.12, right=0.98, wspace=0)
        fig.suptitle('%s' % title, fontsize=12, fontweight='bold')
    else:
        if len(labels) > 0:
            fig.subplots_adjust(bottom=0.15, top=0.88, hspace=0.3, left=0.12, right=0.98, wspace=0)
        else:
            fig.subplots_adjust(bottom=0.15, top=0.95, hspace=0.3, left=0.12, right=0.98, wspace=0)

    ax = fig.add_subplot(111, facecolor='#FFFFEC')

    nsymbols = len(symbols)
    nlabels = len(labels)

    # Iterate over the 'datasets' tuple:
    for item, dataset in enumerate(datasets):
        # assert(type(dataset) is ndarray)
        if dataset.shape[0] < 1:  # check if dataset is empty
            continue
        ncols = dataset.shape[1]
        assert ncols > 1  # check if there are at least 2 columns
        phase = dataset[:, 0]
        mag = dataset[:, 1]

        if ncols > 2:
            magerr = dataset[:, 2]
        else:
            magerr = None

        if nsymbols > item:
            symbol = symbols[item]
            color = None
        else:
            symbol = 'o'
            color = next(ax._get_lines.prop_cycler)['color']

        if nlabels > item:
            label = labels[item]
        else:
            label = None

        if item in fillerr_index:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                base, = ax.plot(phase, mag, symbol, label=label, color=color, zorder=48)
            # Shade the 95% credible interval around the optimal solution.
            ax.fill(np.concatenate([phase.ravel(), phase.ravel()[::-1]]),
                    np.concatenate([mag.ravel() - 1.9600 * magerr,
                                    (mag.ravel() + 1.9600 * magerr)[::-1]]),
                    alpha=.4, fc=base.get_color(), ec='None', zorder=70)
        else:
            ax.errorbar(phase, mag, yerr=magerr, fmt=symbol, label=label, capsize=capsize, color=color)
            if maxphase > 1:
                ax.errorbar(phase + 1, mag, yerr=magerr, fmt=symbol, capsize=capsize, color=color)

    if nlabels > 0:
        plt.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.20),
                   ncol=4, fancybox=True, shadow=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if len(figtext) > 0:
        ax.text(0.05, 1.02, "%s" % figtext, ha='left', va='top', bbox=dict(boxstyle='round', ec='k', fc='w'),
                transform=ax.transAxes)

    plt.xlim(minphase, maxphase)

    if constrain_yaxis_range:
        # The y axis range will be will be optimized for the range datasets[0][1].
        # print(datasets[1][1])
        minmag = np.min(datasets[1][:, 1])
        maxmag = np.max(datasets[1][:, 1])
        magrange = maxmag - minmag
        ax.set_ylim(minmag - magrange / 5., maxmag + magrange / 5.)

    if invert_y_axis:
        plt.gca().invert_yaxis()

    # plt.tight_layout()
    if figsave and (outfile is not None):
        fig.savefig(outfile, format=figformat)
        plt.close(fig)
    else:
        fig.show()

    return None


def extend_phases(p, y, phase_ext_neg=0.0, phase_ext_pos=0.0, sort=False):
    """
    Extend a phase and a corresponding data vector in phase.
    """

    # Extend data vectors in phase:
    neg_ext_mask = (p - 1 > phase_ext_neg)  # select phases in negative direction
    pos_ext_mask = (p + 1 < phase_ext_pos)  # select phases in positive direction

    # Compose new data vectors according to extended phases:
    p_ext = np.hstack((p[neg_ext_mask] - 1, p, p[pos_ext_mask] + 1))
    y_ext = np.hstack((y[neg_ext_mask], y, y[pos_ext_mask]))
    # magerr_ext=np.hstack((results['magerr_binned'][neg_ext_mask], results['magerr_binned'],
    # results['magerr_binned'][pos_ext_mask]))

    if sort:
        # Sort data according to observed phases:
        indx = np.argsort(p_ext)  # indices of sorted ophase
        p_ext_sorted = p_ext[indx]
        y_ext_sorted = y_ext[indx]
        return p_ext_sorted, y_ext_sorted
    else:
        return p_ext, y_ext


def smolec_feh(period, phi31, amp2):
    return -6.125 - 4.795 * period + 1.181 * phi31 + 7.876 * amp2
