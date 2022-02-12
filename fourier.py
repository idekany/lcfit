import sys
import warnings
import numpy as np
import lcfit_utils as ut
import subprocess
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, ExpSineSquared, WhiteKernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, cross_val_score
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from scipy.stats import binned_statistic as binstat
from time import time


def glsper(times: np.ndarray, values: np.ndarray, errors: np.ndarray, glsinputfile: str = 'gls_input.dat'):
    """
    Calls the external code "gls" on the input time series,
    that implements the Generalized Lomb-Scargle algorithm.

    :param times: 1-dimensional numpy.ndarray
    The times of the measurements.

    :param values: 1-dimensional numpy.ndarray
    Values of the measurements.

    :param errors: 1-dimensional numpy.ndarray
    Errors in the measurements.

    :param glsinputfile: string (default: 'gls_input.dat')
    Name of the file to be read by the external code 'gls'.

    :return:
    period: float
    Returned by the GLS algorithm.
    """

    assert len(times.shape) == 1, "parameter `times` must be a rank-1 array"
    assert len(values.shape) == 1, "parameter `values` must be a rank-1 array"
    assert len(errors.shape) == 1, "parameter `errors` must be a rank-1 array"

    assert times.shape == values.shape == errors.shape, "the arrays passed as parameters must have the same shape"

    glsinput = np.vstack((times, values, errors))
    np.savetxt(glsinputfile, glsinput.T)
    # glscmd = "./gls " + glsinputfile + " | awk '{print $3}'"
    # glsout = subprocess.check_output(glscmd, stderr=None, shell=True, universal_newlines=False)
    glsout = subprocess.run(['./gls', glsinputfile], capture_output=True)
    period = 1.0/float(glsout.stdout.split()[2])
    return period


def get_phases(period: float, x: np.ndarray, epoch: float = 0.0, shift: float = 0.0, all_positive: bool = True):
    """
    Compute the phases of a monoperiodic time series.

    :param period: float
    
    :param x: 1-dimensional numpy.ndarray
    The array of the time values of which we want to compute the phases.
        
    :param epoch: float, optional (default=0.0)
    Time value corresponding to zero phase.
        
    :param shift: float, optional (default=0.0)
    Phase shift wrt epoch.
        
    :param all_positive: boolean, optional (default=True)
    If True, the computed phases will be positive definite.
        
    :return:
    phases: 1-dimensional numpy.ndarray
    The computed phases with indices matching x.
    """

    phases = np.modf((x-epoch+shift*period)/period)[0]

    if all_positive:
        phases = all_phases_positive(phases)
    return phases

    
def all_phases_positive(phases: np.ndarray):
    """
    Converts an array of phases to be positive definite.

    :param phases : 1-dimensional numpy.ndarray
    The phases to be modified.
    
    :return: 1-dimensional numpy.ndarray
    Positive-definite version of phases.
    """

    while not (phases >= 0).all():
        phases[phases < 0] = phases[phases < 0] + 1.0
    return phases


def fsum_sin(x: np.ndarray, amps: np.ndarray, phases: np.ndarray,
             intercept: float = 0, period: float = 1, shift: float = 0.0):
    """
    Compute a truncated Fourier sum using sine terms.

    :param x: 1-dimensional ndarray
    Array with the values of the independent variable.

    :param amps:1-dimensional ndarray
    Amplitudes (coefficients) of the sine terms.

    :param phases: 1-dimensional ndarray
    Phases of the sine terms.

    :param intercept: float
    Zero-frequency constant.

    :param period: float
    The main periodicity of the time series.

    :param shift: float
    A shift applied to x.

    :return:
    fourier_sum: 1-dimensional ndarray
    The array with the values of the Fourier sum.
    """

    assert len(x.shape) == 1, "parameter `x` must be a rank-1 array"
    assert len(amps.shape) == 1, "parameter `x` must be a rank-1 array"
    assert len(phases.shape) == 1, "parameter `x` must be a rank-1 array"
    assert amps.shape == phases.shape, "the shapes of `amps` and `phases` must match, got {} and {}."\
        .format(amps.shape, phases.shape)
    order = amps.shape[0]

    arg_vect = 2*np.pi*(x+shift)/period
    result = np.zeros(x.shape[0])

    for i in range(1, order+1):
        result = result + amps[i-1]*np.sin(i*arg_vect+phases[i-1])

    fourier_sum = result + intercept

    return fourier_sum


class ExtendPhases(BaseEstimator, TransformerMixin):
    """
    Transformer class to be used as an element in sklearn Pipeline objects.
    Useful for transforming a padded array of phases with values in [0:1] into an
    extended range by reaassigning appropriate values beyond [0:1].
    Example:
         input array:  [0.7, 0.9, 0.1, 0.2, 0.5, 0.7, 0.9, 0.1, 0.2]
         transformed array: [-0.3, -0.1, 0.1, 0.2, 0.5, 0.7, 0.9, 1.1, 1.2]
    Indended to be combined with the PaddedRepeatedKFold custom CV-splitter class
    in order to create periodic boundary conditions using padded phases during cross-validation.
    """
    def __init__(self):
        # defined for compatibility with sklearn
        pass

    def fit(self, X=None, y=None):
        # defined for compatibility with sklearn
        return self

    def transform(self, X: np.ndarray, y=None):
        """
        Transforms a padded phase array to a support beyond [0,1].

        :param X: 1-dimensional numpy.ndarray
        The input values in [0:1].

        :param y: None
        Exists for compatibility.

        :return: 1-dimensional numpy.ndarray
        The transformed input.
        """

        xx = X.ravel()
        # find the (two) indices where the element in `xx` is smaller than the
        # previous one:
        indices = list(np.where(np.less(xx, np.roll(xx, 1)))[0])
        if len(indices) == 2:
            ind1, ind2 = indices
            # modify the elements of `xx` beyond before and after these indices:
            xx[:ind1] = xx[:ind1] - 1
            xx[ind2:] = xx[ind2] + 1

            return xx.reshape(-1, 1)
        else:
            return X


def extend_phases(X: np.ndarray, y: np.ndarray, ext_neg: float = 0.0, ext_pos: float = 0.0, sort: bool = False):
    """
    Pads an array of phases `X` with values in [0:1] to [ext_neg, ext_pos].
    The array `y` is also padded with its values corresponding to the indices of `X`.

    :param X: 1-dimensional numpy.ndarray
    The phase array with values in [0,1].

    :param y: 1-dimensional numpy.ndarray
    An array with values corresponding to each phase in `X`.

    :param ext_neg: float
    The negative extension of the padding of `X`.

    :param ext_pos:
    The positive extension of the padding of `X`.

    :param sort: bool
    If `True`, the output arrays will be sorted according to the padded version of `X`.

    :return: 1-dimensional numpy.ndarray, 1-dimensional numpy.ndarray
    The transformed versions of `X` and `y`.
    """

    # X and y are expected to be rank-1 arrays

    # Extend data vectors in phase:
    neg_ext_mask = (X - 1 > ext_neg)  # select phases in negative direction
    pos_ext_mask = (X + 1 < ext_pos)  # select phases in positive direction

    # Compose new data vectors according to extended phases:
    X_ext = np.hstack((X[neg_ext_mask] - 1, X, X[pos_ext_mask] + 1))
    y_ext = np.hstack((y[neg_ext_mask], y, y[pos_ext_mask]))
    # magerr_ext=np.hstack((results['magerr_binned'][neg_ext_mask], results['magerr_binned'],
    # results['magerr_binned'][pos_ext_mask]))

    if sort:
        # Sort data according to observed phases:
        indx = np.argsort(X_ext)  # indices of sorted ophase
        p_ext_sorted = X_ext[indx]
        y_ext_sorted = y_ext[indx]
        return p_ext_sorted, y_ext_sorted
    else:
        return X_ext, y_ext


class PaddedRepeatedKFold:
    """
    Custom cross-validation splitter class to be used with sklearn.
    It expects the input array to be a sorted array of phases in the range [0,1].
    For the training indices, it returns a random subset of this array padded to the range of [ext_neg, ext_pos].
    Intended to be used together with the ExtendPhases transformer class in order to create periodic
    boundary conditions using padded phases during cross-validation.
    """
    def __init__(self, n_splits: int = 3, n_repeats: int = 1, ext_neg: float = 0.0, ext_pos: float = 0.0,
                 random_state: int = 1):
        """
        :param n_splits: int
        The number of cross-validation splits.

        :param n_repeats: int
        The number of repeats of the cross-vaidation splits.

        :param ext_neg: float
        The negative extension of the padding of `X`.

        :param ext_pos: float
        The positive extension of the padding of `X`.

        :param random_state: int
        Integer seed for reproducibility.
        """

        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.ext_neg = ext_neg
        self.ext_pos = ext_pos

    def split(self, X, y=None, groups=None):
        """
        Create random training - validation splits.
        The training indices will be padded by repeating the indices of `X` over its range of [ext_neg,ext_pos].

        :param X: 1-dimensional numpy.ndarray
        The array of independent variables.

        :param y: None
        Exists for compatibility.

        :param groups: None
        Exists for compatibility.

        :return: yields a pair of 1-dimensional numpy.ndarray objects
        The index arrays of the training and validation sets.
        """

        for tr_index, val_index in RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats,
                                                 random_state=self.random_state).split(X):
            # X is expected to be rank-2 array, in order for the method to be compatibl with sklearn
            # IMPORTANT: X is expected to be SORTED
            # the index arrays are rank-1

            neg_ext_mask = (X[tr_index, :] - 1 > self.ext_neg).ravel()  # extend X in negative direction
            pos_ext_mask = (X[tr_index, :] + 1 < self.ext_pos).ravel()  # extend X in positive direction

            tr_index_ext = np.hstack((tr_index[neg_ext_mask], tr_index, tr_index[pos_ext_mask]))

            # Sort data according to observed phases:
            # print(tr_index_ext.shape, val_index.shape)
            yield tr_index_ext, val_index

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


class GPRModel:
    """
    Gaussian Process Regressor (GPR) model with periodic kernel for fitting phase-folded monoperiodic time series.
    Based on the sklearn.gaussian_process.GaussianProcessRegressor class.
    """

    def __init__(self, maxn_gpr: int = 200, phase_ext_neg=0, phase_ext_pos=1.2, hparam_optimization: str = 'mle',
                 n_restarts_optimizer: int = 0, n_init: int = 10, n_calls: int = 10,
                 lower_length_scale_bound=0.1, upper_length_scale_bound=10.0, n_jobs=1):
        """

        :param maxn_gpr: int
        The maximum number of input points to be allowed for regression without binning.
        If the length of the `X` array exeeds this number, the input will be binned to `maxn_gpr` points.

        :param phase_ext_neg: float
        The minimum phase of the prediction. At fitting time, the input data will be padded with itself
        to the range of [phase_ext_neg-1, phase_ext_pos+1] to form periodic a boundary condition.

        :param phase_ext_pos:
        The maximum phase of the prediction. At fitting time, the input data will be padded with itself
        to the range of [phase_ext_neg-1, phase_ext_pos+1] to form periodic a boundary condition.

        :param hparam_optimization: str
        The method of hyperparameter optimization. Valid options are 'mle' and 'cv'.
        `mle`: The optimal hyperparameters will be determined by maximizing the log-marginal likelihood.
        `cv`: The optimal hyperparameters will be determined by 10-fold cross-validation and Bayesian optimization,
        employing the skopt API.

        :param n_restarts_optimizer: int
        The number of restarts of the optimizer for finding the kernel’s parameters which maximize
        the log-marginal likelihood. The first run of the optimizer is performed from the kernel’s
        initial parameters, the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds must be finite.
        Note that n_restarts_optimizer == 0 implies that one run is performed.

        :param n_init: int
        Initial calls for the Bayesian hyperparameter optimization (if `hparam_optimization` is `cv`).

        :param n_calls: int
        Number of iterations of the Bayesian hyperparameter optimization (if `hparam_optimization` is `cv`).
        """

        assert hparam_optimization in ('mle', 'cv'), "value of `hparam_optimization` must be `mle` or `cv`"

        self._maxn_gpr = maxn_gpr
        self._phase_ext_neg = phase_ext_neg
        self._phase_ext_pos = phase_ext_pos
        self._gpr = None
        self._gpr_input_phase = None
        self._init_kernel = None
        self._hparam_optimization = hparam_optimization
        self._n_init = n_init
        self._n_calls = n_calls
        self._n_restarts_optimizer = n_restarts_optimizer
        self._noise_level = None
        self.lower_length_scale_bound = lower_length_scale_bound
        self.upper_length_scale_bound = upper_length_scale_bound
        self.n_jobs = n_jobs

    def fit(self, x: np.ndarray, y: np.ndarray, noise_level: float, verbose=True, random_state: int = None):
        """
        Fit a GPR regressor with periodic kernel.
        The periodicity is fixed to 1, it can be used to fit a phase-folded light curve.
        The phase values are repeated beyond [0,1] to form periodic boundary conditions.
        The data are binned to a maximum of _maxn_gpr points to save computation time.

        :param x: 1-dimensional ndarray
        Array with the values of the descriptive variable (phases).

        :param y: 1-dimensional ndarray
        Array with the values of the target variable (magnitudes).

        :param noise_level: float
        Initial noise level parameter for the WhiteKernel.

        :param verbose: boolean
        Turn on/off verbosity.

        :param random_state: int
        Integer seed for reproducibility.

        :return: self
        Returns an instance of self.
        """

        if verbose:
            print("Performing GPR fit...")
        tic = time()

        self._noise_level = noise_level

        # Check if the number of data points exceeds maxn_gpr
        if len(x) > self._maxn_gpr:
            # Bin data to maxn_gpr points:
            gpr_bins = np.linspace(0.0, 1.0, self._maxn_gpr+1)
            gpr_input_mag = binstat(x, y, statistic='mean', bins=gpr_bins).statistic
            gpr_input_phase = (gpr_bins+1./self._maxn_gpr/2.)[:-1]
            nanmask = np.isnan(gpr_input_mag)
            gpr_input_mag = gpr_input_mag[~nanmask]
            self._gpr_input_phase = gpr_input_phase[~nanmask]

        else:
            gpr_input_mag = y
            self._gpr_input_phase = x

        # Sort the input data by phases:
        indx = np.argsort(self._gpr_input_phase)
        self._gpr_input_phase = self._gpr_input_phase[indx]
        gpr_input_mag = gpr_input_mag[indx]

        # We will use a scaled periodic kernel plus white noise:
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * \
            ExpSineSquared(length_scale=1.0, periodicity=1.0,
                           length_scale_bounds=(self.lower_length_scale_bound, self.upper_length_scale_bound),
                           periodicity_bounds="fixed") + \
            WhiteKernel(noise_level=noise_level ** 2)

        self._init_kernel = kernel

        if self._hparam_optimization == 'cv':
            # in this case, we will not optimize (maximize) the log-likelihood:
            optimizer = None
            self._n_restarts_optimizer = 0
        else:
            # optimizer for maximizing the log-ikelihood:
            optimizer = 'fmin_l_bfgs_b'

        # Instantiate sklearn GPR model:
        self._gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=self._n_restarts_optimizer,
                                             normalize_y=False, optimizer=optimizer)

        if self._hparam_optimization == 'cv':
            # define sklearn pipeline:
            if verbose:
                print("GPR Bayesian hyperparameter optimization...")
            steps = list()
            steps.append(('transformer', ExtendPhases()))
            steps.append(('estimator', self._gpr))
            pipeline = Pipeline(steps=steps)

            # define skopt search space:
            search_space = list()
            search_space.append(Real((noise_level / 2.) ** 2, (noise_level * 2) ** 2, 'uniform',
                                     name='estimator__kernel__k2__noise_level'))
            search_space.append(Real(1, 100, 'uniform', name='estimator__kernel__k1__k1__constant_value'))
            search_space.append(Real(self.lower_length_scale_bound,
                                     self.upper_length_scale_bound,
                                     'uniform', name='estimator__kernel__k1__k2__length_scale'))

            # define cv-splitter:
            cv = PaddedRepeatedKFold(n_splits=10, n_repeats=1, ext_neg=self._phase_ext_neg-1.0,
                                     ext_pos=self._phase_ext_pos+1.0, random_state=random_state)

            @use_named_args(search_space)
            def evaluate_model(**params):
                pipeline.set_params(**params)
                # compute CV scores per fold
                # scoring=None --> the regressor's default scorer will be used (in this case, the R2 score)
                scores = cross_val_score(pipeline, self._gpr_input_phase.reshape(-1, 1),
                                         gpr_input_mag.reshape(-1, 1), cv=cv, n_jobs=self.n_jobs,
                                         scoring=None)
                # calculate the mean of the scores
                score = np.mean(scores)
                # convert score to be minimized as the objective
                return 1 - score

            warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')
            result = gp_minimize(evaluate_model, search_space, verbose=False, n_calls=self._n_calls,
                                 n_initial_points=self._n_init, n_jobs=self.n_jobs)

            pipeline.set_params(estimator__kernel__k2__noise_level=result.x[0])
            pipeline.set_params(estimator__kernel__k1__k1__constant_value=result.x[1])
            pipeline.set_params(estimator__kernel__k1__k2__length_scale=result.x[2])

            if verbose:
                print("Best score: {0:.3f}".format(1 - result.fun))
                # print("Best hyperparameters: {0:s}".format(str(result.x)))
                print("GPR noise level: {0:.3f}".format(np.sqrt(result.x[0])))

            # extract tuned regressor from pipeline:
            self._gpr = pipeline.steps[1][1]

        # prepare boundary conditions in the dataset for the final regression:
        phase_gpr, mag_gpr = \
            ut.extend_phases(self._gpr_input_phase, gpr_input_mag,
                             phase_ext_neg=self._phase_ext_neg-1.0, phase_ext_pos=self._phase_ext_pos+1.0,
                             sort=True)

        print("initial noise level = {0:.5f}".format(noise_level ** 2))
        self._gpr.fit(phase_gpr.reshape(-1, 1), mag_gpr.reshape(-1, 1))
        print("final kernel:")
        print(self._gpr.kernel_)

        r2score = self._gpr.score(phase_gpr.reshape(-1, 1), mag_gpr.reshape(-1, 1))

        toc = time()
        if verbose:
            print("   ... completed in {0:.1f} s".format(toc - tic))
            print("GPR R2 score = {0:.3f}".format(r2score))

        return self

    def predict(self, x: np.ndarray, return_std=True):
        """

        :param x: 1-dimensional numpy.ndarray
        Query points where the GP is evaluated.

        :param return_std: bool (default=True)
        If True, the standard-deviation of the predictive distribution
        at the query points is returned along with the mean.

        :return: 1-dim. numpy.ndarray [, 1-dimensional numpy.ndarray]
        Mean of predictive distribution at query points,
        [Standard deviation of predictive distribution at query points.
        Only returned when `return_std` is `True`.]
        """

        if self._gpr is None:
            return None

        if return_std:
            yhat, sigma = self._gpr.predict(x.reshape(-1, 1), return_std=return_std)
            yhat = yhat.ravel()
            return yhat, sigma

        else:
            yhat = self._gpr.predict(x.reshape(-1, 1), return_std=return_std)
            return yhat.ravel()

    def get_fourier_errors(self, fourier_from_gpr: bool = False, n_samples: int = 100, order=3, random_state: int = 1,
                           mean_phase2: float = None, mean_phase3: float = None,
                           mean_phase21: float = None, mean_phase31: float = None,
                           feh_model=None, pca_model=None,
                           period=1.0):
        """
        Computes error estimates in the Fourier parameters by drawing samples from the GPR model, fitting a Fourier
        model to each realization, and obtaining the standard deviations of the resulting Fourier parameters.

        :param fourier_from_gpr: bool
        If `False`, a direct Fourier fit will be computed at each realization.
        If `True`, the Fourier representation of the GPR model will be used instead.
        NOTE: A setting to `True` can significantly slow down the computation. A `False` setting usually yields
        rather similar results.

        :param n_samples: int
        The number of samples to be drawn from the GPR model.

        :param order: int
        The number of Fourier terms to be fitted if `fourier_from_gpr` is `False`. Otherwise a 20-term
        Fourier series will be fitted to the GPR model evaluated over a dense phase grid.

        :param random_state: int
        Integer seed for reproducibility.

        :param mean_phase2: float
        Population mean for the second Fourier phase. If not `None`, the computed phase will be shifted by 2 PI
        to take its closest value to `mean_phase2`.

        :param mean_phase3: float
        Population mean for the third Fourier phase. If not `None`, the computed phase will be shifted by 2 PI
        to take its closest value to `mean_phase3`.

        :param mean_phase21: float
        Population mean for the phi2 - 2 * phi1 Fourier parameter.
        If not `None`, the computed phase will be shifted by 2 PI to take its closest value to `mean_phase21`.

        :param mean_phase31: float
        Population mean for the phi3 - 3 * phi1 Fourier parameter.
        If not `None`, the computed phase will be shifted by 2 PI to take its closest value to `mean_phase31`.

        :param feh_model: object
        An sklearn predictor instance employing a `predict` method. If not `None`, it will be used for predicting
        errors in the [Fe/H] chemical index from the Fourier parameters period, A1, phi31.

        :param pca_model: object
        An sklearn PCA decomposition model instance employing the `transform` method. If not `None`, it will be
        used for predicting errors in the transformed features.

        :param period: float
        The period of the time series.

        :return: dict
        A dictionary with the computed error estimates.
        """

        if self._gpr_input_phase is None:
            print("Error: GPR model has not yet been fitted to data.", file=sys.stderr)
            return

        samples_gpr = self._gpr.sample_y(self._gpr_input_phase.reshape(-1, 1), n_samples=n_samples,
                                         random_state=random_state).reshape(self._gpr_input_phase.shape[0], n_samples)

        if fourier_from_gpr:
            order = 20
        fme = FourierModel(order=order, c_freq_tol=1e-10)
        gpr_sample_icept = np.zeros(n_samples)
        gpr_sample_amp = np.zeros((n_samples, order))
        gpr_sample_pha = np.zeros((n_samples, order))
        gpr_sample_phi21 = np.zeros(n_samples)
        gpr_sample_phi31 = np.zeros(n_samples)
        gpr_sample_feh = np.zeros(n_samples)
        gpr_sample_pca = np.zeros((n_samples, 6))

        for js in range(n_samples):
            # print(js)
            if fourier_from_gpr:
                # First, fit a GPR to the current sample, and then compute the Fourier parameters from the GPR model.
                # NOTE: this can be very slow.
                #       A direct Fourier fit of the samples yields similar results in much less time.
                self.fit(self._gpr_input_phase, samples_gpr[:, js], self._noise_level, verbose=False)
                synmag_sample = self.predict(np.linspace(0.0, 1.0, num=100, endpoint=False), return_std=False)
                fme.set_params({'c_freq_tol': 1e-10, 'period_': 1.0, 'order': 20})
                fme.fit(self._gpr_input_phase, synmag_sample, predict=False)
            else:
                # Fit a Fourier series directly to the current sample:
                fme.fit(self._gpr_input_phase, samples_gpr[:, js], predict=False)
            # print("fit done")
            gpr_sample_icept[js] = fme.intercept_
            gpr_sample_amp[js, :] = fme.amplitudes_

            if mean_phase2 is not None:
                fme.phases_[1] = shift_phase(fme.phases_[1], mean_phase=mean_phase2)

            if mean_phase3 is not None:
                fme.phases_[2] = shift_phase(fme.phases_[2], mean_phase=mean_phase3)

            gpr_sample_pha[js, :] = fme.phases_

            p21 = fme.phases_[1] - 2 * fme.phases_[0]
            if mean_phase21 is not None:
                p21 = shift_phase(p21, mean_phase=mean_phase21)
            gpr_sample_phi21[js] = p21

            p31 = fme.phases_[2] - 3 * fme.phases_[0]
            if mean_phase31 is not None:
                p31 = shift_phase(p31, mean_phase=mean_phase31)
            gpr_sample_phi31[js] = p31

            if feh_model is not None:
                # gpr_sample_feh[js] = ut.smolec_feh(period, p31, fme.amplitudes_[1])
                gpr_sample_feh[js] = feh_model.predict(np.array([[period, fme.amplitudes_[1], p31]]))

            if pca_model is not None:
                gpr_sample_pca[js, :] = \
                    pca_transform(pca_model,
                                  np.array([period, fme.amplitudes_[0], fme.amplitudes_[1], fme.amplitudes_[2],
                                            p21, p31]))

        results = ({'icept_std': np.std(gpr_sample_icept),
                    'A_std': np.std(gpr_sample_amp, axis=0),
                    'Pha_std': np.std(gpr_sample_pha, axis=0),
                    'phi21_std': np.std(gpr_sample_phi21),
                    'phi31_std': np.std(gpr_sample_phi31)})
        if feh_model is not None:
            results.update({'feh_std': np.std(gpr_sample_feh)})
        if pca_model is not None:
            results.update({'pca_feat_std': np.std(gpr_sample_pca, axis=0)})

        return results

    def augment_data(self, phases: np.ndarray, n_aug: int = 10, verbose: bool = True, random_state: int = None):
        """
        Draw samples from a fitted GPR model for data augmentation.

        :param phases: numpy.ndarray
        The queried phase grid.

        :param n_aug: int
        The number of augmented time series to generate.

        :param verbose: bool
        Verbosity parameter.

        :param random_state: int
        Integer seed for reproducibility.

        :return: numpy.ndarray, numpy.ndarray (of shape: (n_phases, n_aug))
        The augmented data queried at the original phase points.
        The augmented data queried at the grid supplied in the `phases` parameter.
        """

        if self._init_kernel is None:
            raise ValueError("Error: GPR model has not yet been fitted to the data.")

        # Sample n_aug time series from the GPR model
        if verbose:
            print("Performing GPR data augmentation...")
        tic = time()
        aug_samples_gpr = \
            self._gpr.sample_y(self._gpr_input_phase.reshape(-1, 1), n_samples=n_aug,
                               random_state=random_state + 1).reshape(self._gpr_input_phase.shape[0], n_aug)

        # Loop through each GPR sample:
        synmag_gpa = np.zeros((len(phases), n_aug))
        for iaug in range(n_aug):
            # extend phases (acts as border condition)
            phase_gpr_aug, mag_gpr_aug = \
                ut.extend_phases(self._gpr_input_phase, aug_samples_gpr[:, iaug],
                                 phase_ext_neg=self._phase_ext_neg - 1.0,
                                 phase_ext_pos=self._phase_ext_pos + 1.0,
                                 sort=True)

            # fit sample with a GPR
            gpa = GaussianProcessRegressor(kernel=self._init_kernel, n_restarts_optimizer=0, normalize_y=False)
            gpa.fit(phase_gpr_aug.reshape(-1, 1), mag_gpr_aug.reshape(-1, 1))
            # evaluate the fit:
            smag = gpa.predict(phases.reshape(-1, 1), return_std=False).ravel()
            synmag_gpa[:, iaug] = smag

        toc = time()
        if verbose:
            print("   ... completed in {0:.1f} s".format(toc - tic))

        return aug_samples_gpr, synmag_gpa


class FourierModel:
    """
    Non-linear Fourier series regressor.
    """

    def __init__(self, period_init: float = 1.0, order: int = 3, c_freq_tol=10.0, loss: str = 'linear',
                 epsilon: float = 0.1,
                 mean_phase2: float = None, mean_phase3: float = None,
                 mean_phase21: float = None, mean_phase31: float = None,
                 tol: float = 1e-8, verbose: bool = False):

        """
        :param period_init: float
        The initial value for the periodicity in the time series.

        :param order: int (default=3)
        The number of Fourier terms to fit.

        :param c_freq_tol: float (default=10)
        Tolerance parameter for the period fit. The trusted region will be defined as
        [ period_init - c_freq_tol * freq_res, period_init + c_freq_tol * freq_res], where
        freq_res is the inverse of the total timespan.

        :param loss: str
        The loss function to be minimized in the regression. Choices are:

        :param epsilon: float
        Epsilon parameter for the Huber loss (if `loss` is set to `huber`).

        :param mean_phase2: float
        Population mean for the second Fourier phase. If not `None`, the computed phase will be shifted by 2 PI
        to take its closest value to `mean_phase2`.

        :param mean_phase3: float
        Population mean for the third Fourier phase. If not `None`, the computed phase will be shifted by 2 PI
        to take its closest value to `mean_phase3`.

        :param mean_phase21: float
        Population mean for the phi2 - 2 * phi1 Fourier parameter.
        If not `None`, the computed phase will be shifted by 2 PI to take its closest value to `mean_phase21`.

        :param mean_phase31: float
        Population mean for the phi3 - 3 * phi1 Fourier parameter.
        If not `None`, the computed phase will be shifted by 2 PI to take its closest value to `mean_phase31`.

        :param tol: float
        Tolerance parameter for the regression's convergence.

        :param verbose: bool
        Verbosity parameter.
        """

        assert loss in ('linear', 'soft_l1', 'huber', 'cauchy', 'arctan'), \
            "invalid loss, valid choices are:"\
            "`linear`, `soft_l1`, `huber`, `cauchy`, `arctan`"
        
        self.period_ = period_init
        self.order = order
        self.c_freq_tol = c_freq_tol
        self.epsilon = epsilon
        self.verbose = verbose
        self.loss = loss
        self.tol = tol
        self.mean_phase21 = mean_phase21
        self.mean_phase31 = mean_phase31
        self.mean_phase2 = mean_phase2
        self.mean_phase3 = mean_phase3
        self.coefs_ = None
        self.intercept_ = None
        self.amplitudes_ = None
        self.phases_ = None
        self.phi21_ = None
        self.phi31_ = None
        self.cost_ = None
        self.results_ = {}
        self.x = None
        self.y = None
        self.ndata = None
        self.prediction_fit_ = None
        self.residual_fit_ = None

    @property
    def period_(self):
        return self._period_

    @period_.setter
    def period_(self, value):
        if value < 0:
            raise ValueError("Period must be positive")
        self._period_ = value

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        if value <= 0 or value > 30:
            raise ValueError("Fourier order must be in the range (0,30]")
        self._order = value

    def set_params(self, params: dict = None):
        """
        Set object parameters for an instance of the FourierModel class.

        :param params: dictionary
        Object parameters and their new values are passed as key:value pairs.
        """

        if params is None:
            params = {}
        assert type(params) is dict, \
            "Error: argument 'params' must be of type dict, got {}".format(type(params), file=sys.stderr)

        for key, value in params.items():
            if key in self.__dict__.keys():
                # use this if there is no relevant property for the variable in 'key'
                setattr(self, key, value)
            else:
                # use this if there IS relevant property for the variable in 'key',
                # thus 'key' does not exist in the object's __dict__
                super(FourierModel, self).__setattr__(key, value)

    def fourier_sum(self, x, coefs, intercept, period=1, shift=0.0):
        """
        Compute a truncated Fourier sum using mixed terms.

        :param x: 1-dimensional ndarray
        Array with the values of the independent variable.

        :param coefs: 1-dimensional ndarray
        The amplitudes (coefficients) of the terms in sine, cosine order.

        :param intercept: float
        Zero-frequency constant.

        :param period: float
        The main periodicity of the time series.

        :param shift: float
        A shift applied to x.

        :return:
        fourier_sum: 1-dimensional ndarray
        The array with the values of the Fourier sum.
        """
        
        order = self.order
        result = np.zeros(x.size)

        arg_vect = 2*np.pi*(x+shift)/period
        
        for i in range(1, order+1):
            result = result + \
                     coefs[i*2-2] * np.sin(i*arg_vect) + \
                     coefs[i*2-1] * np.cos(i*arg_vect)

        fourier_sum = result+intercept

        return fourier_sum

    def residuals(self, params, x, y, weights=None):
        """
        Return the residuals with respect to a Fourier sum.

        :param params: array-like
        The parameters of the Fourier sum:
        params = [period, As1, Ac1, As2, Ac2, ...], where
        As1 and Ac1 are the coeffiients of the sine and cosine terms, respectively.

        :param x: 1-dimensional ndarray
        Array with the values of the independent variable.

        :param y: 1-dimensional ndarray
        Array with the values of the dependent variable.

        :param weights: 1-dimensional ndarray
        Array of weights to be applied to the residuals.

        :return:
        residuals: 1-dimensional ndarray
        Array of the residuals.
        """
        order = self.order
        
        period = params[0]
        coefs = params[1:2*order+1]   # amplitudes of the sin, cos terms
        intercept = params[2*order+1]   # intercept (additive constant)
        
        predictions = self.fourier_sum(x, coefs, intercept, period)
        
        residuals = y - predictions
        
        if weights is not None:
            # compute weighted residuals
            residuals = residuals*weights
        
        return residuals

    def fit(self, x: np.ndarray, y, weights=None, predict=False):
        """
        Regression of a truncated Fourier series to a time series.

        :param x: 1-dim numpy.ndarray
        Input array with the time values.

        :param y:  1-dim numpy.ndarray
        Input array with the measurements at the time values.

        :param weights:  1-dim numpy.ndarray
        Input array with the sample weights.

        :param predict: bool (default=False)
        If `True`, the method will return the regression model's predictions at the points in `x`, along with the
        residuals.

        :return: numpy.ndarray, numpy.ndarray (with shape (n_samples, ))
        If `predict` is `True`, the predcitions at the input `x` values and their residuals are returned.
        """

        self.x = x
        self.y = y

        self.ndata = len(x)
        
        freq = 1.0/self.period_    # input frequency
        freq_res = 1.0/(np.amax(x)-np.amin(x))    # nominal frequency resolution
        freq_hi = freq+self.c_freq_tol*freq_res
        per_lo = 1.0/freq_hi
        freq_lo = freq-self.c_freq_tol*freq_res
        if freq_lo > 0.0:
            per_hi = 1.0/(freq-self.c_freq_tol*freq_res)
        else:
            per_hi = np.inf
        
        median_y = np.median(y)
        y_range = np.amax(y) - np.amin(y)
        
        # set initial parameters and trusted ranges:
        initial_parameters = np.r_[self.period_, np.zeros(2*self.order), median_y]
        # print("initial parameters:")
        # print(initial_parameters)
        lower = np.r_[per_lo, np.ones(2*self.order)*-y_range, -np.inf]
        upper = np.r_[per_hi, np.ones(2*self.order)*y_range, np.inf]

        if weights is not None:
            args = (x, y, weights)
        else:
            args = (x, y)

        regression = least_squares(self.residuals, x0=initial_parameters, args=args,
                                   loss=self.loss, f_scale=self.epsilon, bounds=(lower, upper),
                                   ftol=self.tol, xtol=self.tol, gtol=self.tol)

        theta = regression.x    # array of the fitted parameters
        cost = regression.cost
        
        period_fit = theta[0]              # fitted_period
        coefs_fit = theta[1:2*self.order+1]     # amplitudes of the sin, cos terms
        intercept_fit = theta[2*self.order+1]   # intercept
    
        if self.order > 2:
            amplitudes = np.zeros(self.order)
            phases = np.zeros(self.order)
        else:
            amplitudes = np.zeros(3)
            phases = np.zeros(3)

        for i in range(1, self.order+1):
            amplitudes[i-1] = np.sqrt(coefs_fit[i*2-2]**2 + coefs_fit[i*2-1]**2)
            phases[i-1] = np.arctan2(coefs_fit[i*2-1], coefs_fit[i*2-2])
            # - (i*freq/freq)*np.arctan2(coefs_fit[1], coefs_fit[0])

        if self.mean_phase2 is not None:
            phases[1] = shift_phase(phases[1], mean_phase=self.mean_phase2)

        if self.mean_phase3 is not None:
            phases[2] = shift_phase(phases[2], mean_phase=self.mean_phase3)

        phi21 = phases[1] - 2 * phases[0]
        if self.mean_phase21 is not None:
            phi21 = shift_phase(phi21, mean_phase=self.mean_phase21)

        phi31 = phases[2] - 3 * phases[0]
        if self.mean_phase31 is not None:
            phi31 = shift_phase(phi31, mean_phase=self.mean_phase31)

        self.period_ = period_fit
        self.coefs_ = coefs_fit
        self.intercept_ = intercept_fit
        self.amplitudes_ = amplitudes
        self.phases_ = phases
        self.cost_ = cost
        self.phi21_ = phi21
        self.phi31_ = phi31

        self.results_.update({'A': self.amplitudes_, 'Pha': self.phases_,
                              'coefs': self.coefs_, 'icept': self.intercept_,
                              'phi21': self.phi21_, 'phi31': self.phi31_})
        
        if self.verbose:
            print("order = {}".format(self.order))
            print("N_data = {}".format(self.ndata))
            print("cost = {}".format(cost))
            print("intercept = {}".format(intercept_fit))
            print("P_fit = {}".format(period_fit))
            print("amplitudes = ")
            print(amplitudes)
            print("phases = ")
            print(phases)
        
        if predict:
            self.prediction_fit_ = self.fourier_sum(x, coefs_fit, intercept_fit, period_fit)
            self.residual_fit_ = y - self.prediction_fit_
            
            return self.prediction_fit_, self.residual_fit_

    def predict(self, x, shift=0.0, for_phases=False):
        
        if for_phases:
            period = 1.0
        else:
            period = self.period_
        prediction_fit = self.fourier_sum(x, self.coefs_, self.intercept_, period, shift)

        return prediction_fit

    def compute_results(self, phases: np.ndarray, data: tuple = None, shiftphase=True):
        """
        Computes various quantities from the last fit.

        :param phases: 1-dimensional ndarray
            Grid of phases for computing prediction.

        :param data: tuple of 2 n-dimensional ndarrays (x, y)
            Data for recomputing the predictions and the corresponding residual and SNR.

        :param shiftphase: boolean
            Shift phases so that the zero phase coincides with the phase of the first Fourier term?

        :return: results, dict
            Dictionary of the results.
        """

        if data is None:
            # Number of data points from last fit:
            if self.prediction_fit_ is None:
                raise ValueError("Run fit with predict=True or specify ")
            else:
                # self.ndata = len(self.prediction_fit_)
                synp = self.prediction_fit_
                rstdev = np.std(self.residual_fit_)
        else:
            try:
                x, y = data
            except:
                raise ValueError("Keyword argument 'data' must be a tuple with exactly two elements.")
            self.ndata = len(x)
            synp = self.predict(x)
            residual = y - synp
            rstdev = np.std(residual)

        # Phase of the first Fourier term, we define this as zero phase
        # and use it for phase-aligning the folded light curves.
        phase1 = self.phases_[0] / (2 * np.pi)

        # Predictions computed over a grid of phases:
        if shiftphase:
            shift = -phase1
        else:
            shift = 0.0
        synmag = self.predict(phases, shift=shift, for_phases=True)

        # Total (peak-to-peak) amplitude:
        totamp = np.amax(synmag) - np.amin(synmag)  # total (peak-to-peak) amplitude

        # Signal-to-noise ratio:
        snr = totamp * np.sqrt(self.ndata) / rstdev

        self.results_.update(
            {'ndata': self.ndata, 'tamp': totamp, 'stdv': rstdev, 'snr': snr, 'syn': synmag, 'synp': synp,
             'phase_grid': phases})


def phase_coverage(phases):
    """
    Compute phase coverage of a periodic time-series.
    
    Phase coverage = 1 - max(phase gap).
    Input: phases  [array_like]  -  vector of phase values [0:1].
    Output: phcov  [float]  -  phase coverage.
    """
    assert (phases >= 0).all(), "In function phase_coverage: input array contains negative value."

    phases_sorted = np.sort(phases)
    phases_sorted = np.append(phases_sorted, phases_sorted[0]+1)
    phase_differences = phases_sorted-np.roll(phases_sorted, 1)
    phasecov = 1.0 - np.amax(phase_differences[1:])
    
    return phasecov


def shift_phase(phase, mean_phase=5.9):

    shifted_phase = phase-np.floor((phase-mean_phase+np.pi)/2.0/np.pi)*2.0*np.pi
    # shifted_phase = phase-np.floor((phase-mean_phase)/2.0/np.pi)*2.0*np.pi

    return shifted_phase


def pca_transform(pca_model, results):
    """
    Applies a pre-trained PCA transformation on the vector: [period, A1, A2, A3, phi21, phi31]
    :param pca_model: a PCA model instance object (including the standard scaler)
    :param results: dictionary or np.ndarray
    dictionary containing the results of the Fourier decomposition (output of FourierModel.compute_results), or
    np.ndarray containing the following Fourier parameters: [period, A1, A2, A3, phi21, phi31]
    :return: np.ndarray
    updated with the vector 'pca_feat' containing the PCA-transformed features
    """

    assert type(results) == dict or type(results) == np.ndarray

    if type(results) == dict:
        input_vector = np.array([results['period'], results['A'][0], results['A'][1], results['A'][2],
                                results['phi21'], results['phi31']])
    else:
        input_vector = results

    pca_feat = pca_model.transform(input_vector.reshape(1, -1)).flatten()
    return pca_feat


def phase_roll(synmag, adjust=None, phi1=None):
    
    assert adjust == 'mean' or adjust == 'a1' or adjust is None

    nsyn = len(synmag)
    phas = np.linspace(0, 1-1.0/nsyn, nsyn)
    
    # first, detect phase of minimum brightness, ...
    ind_max = np.argmax(synmag)          # index of minimum brigtness
    ind_min = np.argmin(synmag)          # index of minimum brigtness
    max_mag_phase = phas[ind_max]           # phase of maximum magnitude (= minimum brightness)
    min_mag_phase = phas[ind_min]           # phase of maximum magnitude (= minimum brightness)
    phase_a1 = phi1 / (2*np.pi)
    # while phi1<0:                                       # make sure that phi1 is positive
    #   phi1=phi1+2.0*np.pi
    print("phi1 = {}".format(phi1))
    print("phase_a1 = {}".format(phase_a1))
    ind_phase_a1 = int(np.round(phase_a1*nsyn))
    print("ind_phase_a1 = {}".format(ind_phase_a1))
    
    phasediff_min_max = max_mag_phase - min_mag_phase
    
    # Adjust zero phase to max magnitude (minimum brightness):
    synmagm = np.roll(synmag, -1*ind_max)

    # ... then detect first phase of mean magnitude after minimum brightness
    meanmag = np.mean(synmag)    # mean mag
    ind_mean = np.argmin(np.abs(synmagm[0:nsyn/2]-meanmag))
    mean_mag_phase = phas[ind_mean]

    if adjust == 'mean':
        synmagr = np.roll(synmagm, -1*ind_mean)
    elif adjust == 'a1':
        synmagr = np.roll(synmag, ind_phase_a1)
    else:
        synmagr = synmag

    mean_mag_phase = mean_mag_phase + max_mag_phase
    
    return synmagr, meanmag, phasediff_min_max, mean_mag_phase, phase_a1
