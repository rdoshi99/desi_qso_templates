import numpy as np
import astropy
from astropy.table import Table
from scipy.signal import savgol_filter
import fitsio




def enough_obs(fluxes, ivars, min_spectra=50):
    """
    TODO: docstring
    """
    num_ivars_for_wavelength = np.sum(ivars>0, axis=0)
    nonzero_idx = num_ivars_for_wavelength > min_spectra
    fluxes = fluxes[:, nonzero_idx]
    ivars = ivars[:, nonzero_idx]
    
    return fluxes, ivars


def smooth(flux, n=31):
    """
    Smooths the noise from a given spectrum using the scipy function savgol_filter
    Args:
        flux : an array of the flux values for a single spectrum
    Options:
        n : 
    """
    return savgol_filter(flux, n, polyorder=3)


def plot_chisq_dist(chisq_dist, hist=False):
    """
    Plot the distribution of the chisq values
    Args:
        chisq_dist : an array of chi-sq values each corresponding to a spectrum
    Options:
        hist : choose whether to display the distribution as a histogram (True) or 
                in order of spectra (default=False)
    """
    if hist:
        pyplot.hist(chisq_dist);
    else:
        plot(np.arange(len(chisq_dist)), chisq_dist);

    
def read_processed(fileloc, flux_file, ivar_file):
    """
    Reads processed fluxes, ivars from disk
    Returns fluxes and ivars 2D arrays
    
    Args:
        fileloc : location of the processed files
        flux_file : name of the processed fluxes pkl file
        ivar_file: name of the processed ivars pkl file
    """
    with open(fileloc + flux_file,"rb") as file:
        f = pickle.load(file)

    with open(fileloc + ivar_file,"rb") as file:
        i = pickle.load(file)
                    
    return f, i



def chisq_per_spectrum(num, fluxes, ivars):
    '''
    Generates the chisquared value for a particular SPECTRUM in FLUXES
    
    Args:
        spectrum : a given spectrum
        fluxes : a 2D array of processed spectra
        ivars : a 2D array of corresponding processed ivars
    '''
    spectrum = fluxes[num]
    ivar_arr = ivars[num]
    start_idx = np.flatnonzero(spectrum)[0]
    end_idx = np.flatnonzero(spectrum)[-1]
    
    f_means = np.average(fluxes, weights=ivars, axis=0)
    chi_terms = (spectrum-f_means)**2 * ivar_arr
    chi_stat = sum(chi_terms)
    return chi_stat



def calc_chisq_dist(fluxes, ivars):
    """
    Generates an array of chisquared stats for each spectrum in FLUXES
    
    Args:
        fluxes : a 2D array of processed spectra
        ivars : a corresponding 2D array of processed ivars
    """
    mean_flux = np.average(fluxes, weights=ivars, axis=0)
    chi_dist = np.sum((fluxes - mean_flux)**2 * ivars, axis=1)
    return chi_dist


def remove_outliers(fluxes, ivars, chisq_dist, cutoff_val, qsocat):
    """
    Removes spectra from fluxes with above CUTOFF_VAL of chi-squared values
    Returns 2D arrays (fluxes and ivars) without those outlier spectra and a new chi_sq dist
    Args:
        fluxes : a 2D array of processed spectra
        ivars : a corresponding 2D array of processed ivars
        chisq_dist : a distribution of the chi-squared statistics for each spectrum
                     corresponding to the fluxes and ivars arrays
        cutoff_val : the chisq_stat value above which poor spectra are removed
        qsocat : a quasar catalog of the spectra recorded in fluxes
    """
    #- Removes any ivars with < 80% zero values over the relevant wavelength range
    def zerod_ivars(arr):
        start, end = np.nonzero(arr)[0][0], np.nonzero(arr)[0][-1]
        vals = arr[start:end]
        if (np.count_nonzero(vals) < .8*len(vals)):
            return 0
        return 1

    # keep_specs = np.apply_along_axis(zerod_ivars, axis=1, arr=ivars)
    # fluxes = fluxes[keep_specs]
    # ivars = ivars[keep_specs]
    # qsocat = qsocat[keep_specs]
    
    #- Determines which of the spectra are beyond the cutoff
    outlier_bools = chisq_dist >= cutoff_val
    outlier_nums = chisq_dist[outlier_bools]
    
    print('removing {} bad spectra with chi-sq values > {:.3f}'.format(len(outlier_nums), cutoff_val))
    
    #- Removes the outliers from fluxes, ivars, qsocat
    fluxes = fluxes[~outlier_bools]
    ivars_no_outliers = ivars[~outlier_bools]
    qsocat = qsocat[~outlier_bools]
        
    new_dist = calc_chisq_dist(fluxes_no_outliers, ivars_no_outliers)
    
    return np.array(fluxes_no_outliers), np.array(ivars_no_outliers), new_dist, qsocat



def outlier_detection(fluxes, ivars, qsocat):
    """
    Iterates over fluxes and ivars until all outliers removed (when all outliers are within 2SD of the mean)
    Args:
        fluxes : a 2D array of processed spectra
        ivars : a corresponding 2D array of processed ivars
        qsocat : a quasar catalog of the spectra recorded in fluxes
    Options:
        percentile_cutoff : the percent of poor spectra to remove (default=3)    
    """
    #- Generates an original chi-squared distribution, standard deviation and mean
    chisq_dist = calc_chisq_dist(fluxes, ivars)
    
    #- Removes any negative chisq values
    bad_chisq = chisq_dist < 0
    chisq_dist = chisq_dist[~bad_chisq]
    fluxes = fluxes[~bad_chisq]
    ivars = ivars[~bad_chisq]
    qsocat = qsocat[~bad_chisq]
    
    
    #- Define a cutoff function based on the distribution
    cutoff_func = lambda dist: 3*np.mean(dist)
    
    #- Run at least one iteration of removing outliers
    first_cutoff = cutoff_func(chisq_dist)
    orig_len = len(fluxes)
    fluxes, ivars, chisq_dist, qsocat = remove_outliers(fluxes, ivars, chisq_dist, first_cutoff, qsocat)

    count=1
    while (len(fluxes) < orig_len and count < 3):
        print('iteration {}'.format(count))
        fluxes, ivars, chisq_dist, qsocat = remove_outliers(fluxes, ivars, chisq_dist, cutoff_func(chisq_dist), qsocat)
        count += 1
        
    return fluxes, ivars, chisq_dist, qsocat
