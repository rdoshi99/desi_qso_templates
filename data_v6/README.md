The files in this data_v6 directory include the following and generate the following data:

### 00_qsoproc.ipynb
Reads and selects spectra from the catalog

Nov. 27, 2020
Run in DESI 19.2 environment on Cori

--> wrote to /selected_spectra folder

Includes:
- fluxes.pkl,
- ivars.pkl,
- rest_loglams.pkl, 
- kept_idxs.pkl (from the original SDSS catalog),
- skipped_idxs.pkl, and 
- qsocat0.fits (which corresponds to qsocat[kept_idxs] and has a bijective mapping with fluxes)

Properties:
- rest_offset = 2.6
- nbins = 14000
- min_per_lam=50
- count is 27723

Notes:
- these are unnormalized spectra


### 01_normalize.ipynb
Normalizes /selected_spectra by grouping into subsets of 5 and normalizing spectra to the mean flux of this group

Includes:
- fluxes, ivars, wavelengths, rest_loglams, qsocat
- count is 27001
- NOTE: may have to change min_per_lam because doubling number of HMF vectors. TBD.

--> wrote to /cleaned_spectra folder
Includes:
- fluxes_clean.pkl,
- ivars_clean,pkl,
- qsocat_clean.pkl,
- restlams_clean.pkl, and 
- wavelengths_clean.pkl

Properties:
- rest_offset = 2.6
- nbins = 13511
- min_per_lam = 50
- count = 27001


### 02_outliers.ipynb
Removes outliers in 3 iterations using chi-squared statistic

Includes:
- fluxes, ivars, wavelengths, rest_loglams, qsocat
- count is 27001
- NOTE: may have to change min_per_lam because doubling number of HMF vectors. TBD.

--> wrote to /cleaned_spectra folder
Includes:
- fluxes_clean.pkl,
- ivars_clean,pkl,
- qsocat_clean.pkl,
- restlams_clean.pkl, and 
- wavelengths_clean.pkl

Properties:
- rest_offset = 2.6
- nbins = 12938
- min_per_lam = 50
- count = 23396


### 03_hmf.ipynb
Reads normalized and outlier-free files from /cleaned_spectra
- fluxes, ivars, rest_loglams, wavlengths, qsocat

--> wrote to /hmf_templates folder
Includes:
The same files: 
- fluxes.pkl, ivars.pkl, qsocat.pkl, restlams.pkl, wavelengths.pkl, and qsocat,fits

Model files:
- model_V.pkl,
- model_C.pkl, and 
- model_hmf.pkl
- C and V matrices at each iteration

Properties:
- rest_offset = 2.6
- nbins = 12938
- min_per_lam = 50
- count = 23396
- vectors: 10
- model.shape = (23396, 12938)


### 04_empca.ipynb
Reads HMF files from /hmf_templates
- copies fluxes, ivars, rest_loglams, wavlengths, qsocat over
- reads in model_hmf, copies over along with model_C, model_V

Runs empca on model_hmf matrix
--> wrote to /final_data_model folder
Includes:
The same files: 
- fluxes.pkl, ivars.pkl, qsocat.pkl, restlams.pkl, wavelengths.pkl, and qsocat.fits
- model_hmf.pkl, model_C.pkl, and model_V.pkl
New model files:
- final_model.pkl
- 8_eigvecs.pkl
- final_model_8_eigvecs.fits

Properties:
- rest_offset = 2.6
- nbins = 12938
- min_per_lam = 50
- count = 23396
- model_hmf.shape = (23396, 12938)
- kept 8 eigenvectors


