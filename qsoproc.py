import desispec
import desitarget
import desimodel.io
fiberpos = desimodel.io.load_fiberpos()

import numpy as np
import astropy
from astropy.table import Table
from scipy.signal import savgol_filter
import fitsio
import pickle



def smooth(flux, n=31):
    """
    Smooths the noise from a given spectrum using the scipy function savgol_filter
    Args:
        flux : an array of the flux values for a single spectrum
    Options:
        n : 
    """
    return savgol_filter(flux, n, polyorder=3)

    
def plot_spectra(nums, fluxes, alpha=0.7, bin_size=0.0001):
    """
    Plots NUMS spectra from FLUXES
    Args:
        nums : an array specifying which spectra to plot
        fluxes : a 2D array where each element is an array of the flux values for a given spectrum
    Options:
        alpha
        bin_size
    """
    if len(nums) == 0:
        return
    num_xbins = len(fluxes[0])
    for i in nums:
        #- smooths out the ith spectrum
        smoothed = smooth(fluxes[i])
        plot(np.arange(num_xbins)*bin_size, smoothed, alpha=alpha)

    #- formatting
    # plt.ylim(-100, 100)
    # plt.xlim(0.95, 1.2)
        
    
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
        

class ProcQSO:
    
    def __init__(self, qsodir):
        """
        Initialize with a base directory from which to read in QSO catalog files.
        """
        self.basedir = qsodir
        self.fluxes = None
        self.ivars = None
    

    def read_spectrum(plate, mjd, fiber):
        """
        Reads in spectra as astropy Tables
        Args:
            plate
            mjd
            fiber
        """
        basedir = '/global/project/projectdirs/cosmo/data/sdss/dr14/eboss/spectro/redux/v5_10_0/spectra'
        specfile = '{basedir}/{plate}/spec-{plate}-{mjd}-{fiber:04d}.fits'.format(
            basedir=basedir, plate=plate, mjd=mjd, fiber=fiber)

        return Table(fitsio.read(specfile, 'COADD', columns=['flux', 'ivar', 'and_mask', 'loglam']))

    
    def add_rest_loglams(spectrum, z):
        """
        Adds the log of the rest wavelengths for a given spectrum
        Args:
            spectrum : the number of a given spectrum used to obtain the redshift from the QSO catalog
            z : the redshift of spectrum
        Returns:
            Astropy table object of spectrum with added logarithmic rest wavelengths
        """    
        #- defines an array of the logs of the observed wavelengths for a spectrum
        loglams = np.array(spectrum['loglam'])
        #- calculates the rest-frame wavelengths on a logarithmic scale
        rest_loglams = loglams - np.log10(1+z)

        #- adds a truncated rest_loglams column to the table of the given spectrum
        spectrum['rest_loglams'] = np.around(rest_loglams, 4)
        return spectrum

    
    def rest_loglam_offset(spectrum, bin_size=0.0001, rest_loglam_base=2.9):
        """
        Bins the logarithmic rest wavelengths of the spectra into enumerated offset bins
        Args:
            spectrum : an array of fluxes for a single spectrum
        Options:
            bin_size
            rest_loglam_base

        Returns:
            Astropy table object of spectrum with added offset bins
        """
        rest_loglams = np.array(spectrum['rest_loglams'])
        #- constructs an array of differences between the rest wavelengths and the base rest wavelength
        rest_loglam_diffs = rest_loglams - rest_loglam_base
        #- counts the number of bins between each observed wavelength and the base
        num_bins_offset = rest_loglam_diffs / bin_size
        #- adds an offset_bins column to the spectrum table
        spectrum['offset_bins'] = np.round(num_bins_offset).astype(int)
        return spectrum    

    
    def construct_arrays(spectrum, num_bins=12000):
        '''
        Constructs flux and ivar arrays for a given spectrum
        Assigns flux values where and_mask != 0 to a weight of 0 in ivar

        Returns:
            flux and ivar arrays
        '''
        #- creates an empty array of zeros
        flux_array = np.zeros(num_bins)
        ivar_array = np.zeros(num_bins)

        #- for each (offset_bin, flux) pair, add it to the 2D array
        #- likewise for ivar
        for i in range(len(spectrum['offset_bins'])):
            flux_array[spectrum['offset_bins'][i]] = spectrum['flux'][i]

            #- ensure that any flux values where and_mask != 0 is weighted to value 0
            if spectrum['and_mask'][i]:
                ivar_array[spectrum['offset_bins'][i]] = 0
            else:
                ivar_array[spectrum['offset_bins'][i]] = spectrum['ivar'][i]

        return flux_array, ivar_array    
    

    def normalize(fluxes, ivars, spec, i_array, num_normalized):
        """
        For each spectrum, normalizes the flux with respect to the other spectra
        - All transformations to the fluxes are also applied to the inverse variances

        Args:
            fluxes : a 2D array where each element corresponds to an array of the flux values for a different spectrum
            ivars  : a 2D array where each element corresponds to an array of the ivar values for a different spectrum
            spec : a single spectrum (array of fluxes) to be normalized
            i_array : the corresponding ivars to SPEC
            num_normalized : the number of spectra already normalized thus far
        """
        #- get bounds
        nonzeros = np.flatnonzero(spec)
        start_spec = nonzeros[0]
        end_spec = nonzeros[-1]

        mean_spec = np.sum(spec)/(end_spec-start_spec)

        agg_sum = np.sum(fluxes[:num_normalized], axis=0)
        num_bins = int((end_spec - start_spec) * num_normalized)

        agg_mean = np.sum(agg_sum) / num_bins

        #- get scale factor
        scale_factor = mean_spec / agg_mean

        if scale_factor == 0:
            print('ERROR: division by 0 for the {}th spectrum'.format(num_normalized), flush=True)
            #- TODO: make sure this flags such un-normalized spectra
            return

        #- normalize by dividing single spectrum flux by scale factor
        spec = spec / scale_factor
        #- apply all transformations to the ivars
        i_array = i_array / scale_factor

        return spec, i_array

    
    def proc_pipeline(num_spectra=np.linspace(0, 500000, num=10000)):
        """
        Iterates over all spectra, process and noramlizes data, stores in 2D arrays 
        as class variables
        
        Options:
            num_spectra : an array of the indices of the desired quasars from the catalog
                          default is 10,000 spanning the range of 500,000 spectra
            
        Returns fluxes, ivars 2D arrays
        """
        num_spectra = np.unique(num_spectra.astype(int))

        qsocat = Table(fitsio.read(
            '/global/project/projectdirs/cosmo/data/sdss/dr14/eboss/qso/DR14Q/DR14Q_v4_4.fits', 1))
        qsocat.sort('Z')

        #- store in 2D arrays
        fluxes, ivars = [], []

        #- read one spectrum at a time and normalize
        count = 0
        for i in num_spectra:
            try:
                #- read in spectrum and add offset bins
                spec = read_spectrum(qsocat['PLATE'][i], qsocat['MJD'][i], qsocat['FIBERID'][i])
                spec = add_rest_loglams(spec, qsocat['Z'][i])
                spec = rest_loglam_offset(spec)

                #- normalize spec
                f_array, i_array = construct_arrays(spec)

                if count != 0:
                    f_array, i_array = normalize(fluxes, ivars, f_array, i_array, count)

                fluxes.append(f_array)
                ivars.append(i_array)

                count += 1
            except:
                pass

        print('count is ' + str(count))
        plot_spectra(np.arange(count), fluxes)
        
        self.flxues = fluxes
        self.ivars = ivars

        return np.array(fluxes), np.array(ivars)
    
    
    def remove_outliers(self, num_vectors=25):
        """
        Removes the outlier spectra from the processed fluxes and outliers
        Gets the 2D arrays in shape for empca processing
        Options:
            num_vectors : a minimum number of nonzero ivar vectors necessary 
                          for each wavelength
        """
        keep = np.sum(self.ivars > 0, axis=0) > 5*num_vectors
        if sum(keep) == 0:
            print('Data must have wavelengths with at least {} spectra'.format(5*num_vectors))
            return
        self.fluxes = fluxes[:, keep]
        self.ivars = ivars[:, keep]

    
    def save_processed(self, name='', savedir=''):
        """
        Saves processed fluxes, ivars to disk
        Options:
            name : specify appended name of files to save, default is fluxes.pkl, ivars.pkl
            savedir : specify directory to save, default is location of this python file
        """        
        with open("{}fluxes{}.pkl".format(savedir + '/', name),"wb") as file:
            pickle.dump(self.fluxes,file)
        print('saved fluxes to ' + savedir + 'fluxes' + name + '.pkl')
    
        with open("{}ivars{}.pkl".format(savedir + '/', name), "wb") as file:
            pickle.dump(self.ivars, file)
        print('saved ivars to ' + savedir + 'ivars' + name + '.pkl')
        
            
