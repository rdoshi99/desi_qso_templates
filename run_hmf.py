def hmf_weighted(fluxes, ivars, j, num_iter=20, load=False, filestring_a='', filestring_b='', curr_iter=0):
    """
    TODO: docstring
    """
    # ensures that the number of nonzero observations must be 5x 
    # the number of vectors, j=nvec
    fluxes, ivars = enough_obs(fluxes, ivars, j, mult_factor=10)
    
    if not load:
        S = fluxes.T # with dimensions S[nwave, nspec]
        W = ivars.T # the weights matrix with dimensions W[nwave, nspec]
        V = np.random.rand(S.shape[0], j)
        C = np.zeros((j, S.shape[1]))

    else:
        if (not filestring_a) or (not filestring_b):
            print('error: please pass in the correct filestrings to load V and C model components')
            return            
        V, C = load_arrays(filestring_a, filestring_b)
        S = fluxes.T # with dimensions S[nwave, nspec]
        W = ivars.T # weights matrix with dims W[nwave, nspec]

    # track chi stat over time
    chistats = np.zeros((num_iter,))
    
    # iterate
    i = curr_iter
    while(i < num_iter): #TODO: Legit Stopping condition
        C = solveCoeffs(S, W, V, C)
        V = solveEigvecs(S, W, V, C)
        
        # evaluate model
        chi_stat = eval_model(S, W, V, C)
        print('chi stat is {}'.format(chi_stat), flush=True)
        chistats = np.append(chistats, chi_stat)

        # dump files to disk
        with open("finaldata_uniform/model_V_iter_{}.pkl".format(i),"wb") as file:
            pickle.dump(V,file)
    
        with open("finaldata_uniform/model_C_iter_{}.pkl".format(i), "wb") as file:
            pickle.dump(C, file)
        
        i+=1
    C = solveCoeffs(S, W, V, C) # solve coeffs one last time
    return V, C, chistats
    

def solveCoeffs(S, W, V, C):
    """
    TODO: docstring
    """
    A = np.zeros((C.shape[1], C.shape[0], C.shape[0]))    
    b = np.zeros((C.shape[1], C.shape[0]))
    
    output_cached_mat = np.zeros((C.shape[0], V.shape[0]))
    
    for i in range(C.shape[1]): # iterate over all rows of S
        spec_i = S[:, i]
        D = np.diag(W[:, i]) # diagonal matrix of weights corresponding to spec_i
        
        # V^T * D * spec_i = (V^T * D * V) * coeff[:, i]
        # solver determines x where Ax = b, solve for the coefficients column that corresponds to spec_i
        np.dot(V.T, D, out=output_cached_mat)
        np.dot(output_cached_mat, V, out=A[i])
        np.dot(output_cached_mat, spec_i, out=b[i])
    
    C = np.linalg.solve(A, b).T
    return C


def solveEigvecs(S, W, V, C):
    """
    TODO: docstring
    """
    A = np.zeros((V.shape[0], V.shape[1], V.shape[1]))
    b = np.zeros((V.shape[0], V.shape[1],))
    output_cached_mat = np.zeros((V.shape[1], C.shape[1]))
    
    for lam in range(V.shape[0]): # iterate over all wavelengths
        wave_lam = S[lam, :]
        D = np.diag(W[lam, :])
        
        # C * D * wave_lam = (C * D * C.T) * eigvec[:, j]
        # solver determines x where Ax = b
        np.dot(C, D, out=output_cached_mat)
        np.dot(output_cached_mat, C.T, out=A[lam])
        np.dot(output_cached_mat, wave_lam.T, out=b[lam])
    
    V = np.linalg.solve(A, b) # V[lam] = V[lam, :]
    return V


def enough_obs(fluxes, ivars, j, mult_factor=10):
    """
    TODO: docstring
    """
    #- Remove ivars with all zero columns... (pre-processing should have fixed this!)
    idxs = np.sum(ivars, axis=1) == 0
    ivars = ivars[~idxs]
    fluxes = fluxes[~idxs]
    
    print('no more 0-valued ivar spectra:')
    print(sum(np.sum(ivars, axis=1) == 0))

    num_ivars_for_wavelength = np.sum(ivars>0, axis=0)
    nonzero_idx = num_ivars_for_wavelength > mult_factor*j
    fluxes = fluxes[:, nonzero_idx]
    ivars = ivars[:, nonzero_idx]
    
    #- Removes any ivars with < 80% nonzero values over the relevant wavelength range
    
    return fluxes, ivars


def eval_model(S, W, V, C):
    """
    Evaluates the model by computing Delta = S - VC 
    and then calculating Chi2 = SUM[W(lam, i) * Delta(lam, i)**2] over all wavelengths lam and spectra i
    """
    Delta = S - np.dot(V, C)
    D_squared = np.multiply(Delta, Delta)
    chi_stat = np.multiply(W, D_squared)
    return np.sum(np.sum(chi_stat))
    

def load_arrays(filestring_V, filestring_C):
    """
    TODO: docstring
    """
    with open(filestring_V, "rb") as file:
        V = pickle.load(file)
    with open(filestring_C, "rb") as file:
        C = pickle.load(file)
    return V, C
    
    
