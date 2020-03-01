# desi_qso_templates

This module provides a straightforwad method to generate quasar object templates for the DESI survey based on a principal component analysis method, `empca` (@sbailey).

Steps to develop templates:
1) Import and initialize the `ProcQSO` class from the qsoproc submodule.
2) SKIP this step for default processing of 10,000 spectra spanning the whole observed wavelength range. Determine the indices of the spectrum data to be processed from the catalog and store these in an array.
   (This module assumes the catalog is hosted on a public access NERSC directory. To change the read-in location, see the hardcoded `basedir` and `specfile` variables in `qsoproc.read_spectrum`.)
3) Call the `proc_pipeline` function within the ProcQSO class on the default or given spectra to be processed.
4) Call the `remove_outliers ` function within the class to remove poor spectrum data
4) Run the `save_processed` function within the class to write the generated flux and ivar 2D arrays to disk
5) Follow the steps for processing in `empca` linked here: https://github.com/sbailey/empca/blob/master/README.md.


It also contains a WIP notebook with an automated process of Heteroscedastic matrix factorisation (HMF) as it applies to quasar templates. In the future, these methods will be generalized for any application and the results of finding eigenvectors from HMF-factored templates will be compared to generating a model simply from `empca`.
