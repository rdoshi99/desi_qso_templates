# desi_qso_templates

This module provides a straightforward method to generate quasar object templates for the DESI survey based on a two-stage method involving heteroscedastic matrix factorization (HMF) and principal component analysis (PCA). 

HMF is a probabilistic factor analysis method for performing dimensionality reduction on high dimensional data when the application is not agnostic toward non-uniform observational uncertainties. This models the spectrum of each observed object as a sum of `k` linear components in terms by jointly learning the coefficient vector and the basis spectra. The error term is assumed to be drawn from a zero-meaned Gaussian distribution. The HMF method seeks to minimize the scalar objective function, chi-squared. More information: https://iopscience.iop.org/article/10.1088/0004-637X/753/2/122/pdf

PCA then retrieves orthogonal components from the learned basis vectors and weights. The `empca` module (@sbailey) provides an iterative method for solving PCA while properly weighting the data.

Example iPython notebooks to develop templates are provided in the /data_v6 directory.

Steps to develop templates include:
1) Processing `n` spectra spanning the whole observed wavelength range by reading from a hardcoded `basedir` that assumes the catalog is hosted on a public access NERSC directory
2) Normalizing the selected spectra by aggregating into subsets of 5 and standardizing to the mean flux of this group
3) Remove the outliers in 3 iterations using the chi-squared statistic and a tuned cut-off
4) Run the implemented HMF method to learn the weight coefficients and basis vectors
5) Perform PCA on the learned model by following the directions in `empca` linked here: https://github.com/sbailey/empca/blob/master/README.md.
