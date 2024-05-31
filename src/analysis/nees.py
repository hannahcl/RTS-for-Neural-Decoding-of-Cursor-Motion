import numpy as np
from scipy.stats.distributions import chi2

def compute_NEES(estimate, true_mean, cov):

    estimation_error = estimate - true_mean
    EES = estimation_error.T@estimation_error
    NEES = estimation_error.T@np.linalg.inv(cov)@estimation_error

    return EES, NEES

def compute_nees_bounds():
    lower_bound = chi2.ppf(0.025, df=6)
    upper_bound = chi2.ppf(0.975, df=6)

    return lower_bound, upper_bound