import pdb
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_array
from scipy.stats import poisson
from mc_aux import getNxtStateIdx, getIdxForState, \
    getStateFromIdx, numStates, sparseToDict, dictToSparse, isAperiodic, isIrreducible

# sparse representation of a probability transition matrix, in scipy csr format

def sparseSteadyStateDTMC(sP, diag):
    """
    Compute the stationary distribution v of a finite irreducible
    discrete-time Markov chain, where

        v = v @ P
        sum(v) = 1

    Parameters
    ----------
    sP : sparse array format of
        row-stochastic transition probability matrix.
    diag: diagonal elements of sP

    Returns
    -------
    v : ndarray, shape (n,)
        Stationary probability vector.

    Raises
    ------
    ValueError
        If P is not square, contains invalid probabilities, or its rows
        do not sum to one.
    """

    rows, cols = sP.shape
    if rows != cols:
        raise ValueError("P must be a square matrix.")

    n = rows

    # check that each row sums to 1.0
    row_sum = [0.0]*n
    rowIdx, colIdx, V = sparse.find(sP)

    for idx in range(0, len(V)):
        i = rowIdx[idx]; j= colIdx[idx]; pr = V[idx]
        if i==j:
            row_sum[i] += diag[i]
        else:
            row_sum[i] += pr

    for idx in range(0, n):
        if not np.isclose(row_sum[idx], 1.0, atol=1e-12):
            raise ValueError(f"sum of probabilities for state {i} is not 1.0") 

    # create A = P.T - I
    # represent the sparse matrix as a dictionary
    Bdict = sparseToDict(sP, diag)

    # subtract 1 from diagonal
    for row in range(0, n):
        Bdict[row][row] -= 1.0

    # make the last column all ones
    for row in range(0, n):
        Bdict[row][n-1] = 1.0

    A, diag = dictToSparse(Bdict, transpose=True)

    b = np.zeros(n)
    b[-1] = 1.0

    # exact sparse solver
    v = spsolve(A, b)

    # Remove insignificant roundoff errors and renormalize.
    v[np.abs(v) < 1e-15] = 0.0

    if np.any(v < -1e-10):
        raise RuntimeError(
            "The computed stationary distribution has negative components."
        )

    v = np.maximum(v, 0.0)
    v /= v.sum()

    return v

def steadyStateDTMC(P):
    n, _ = P.shape
    try:
        m_format = P.format
    except:
        m_format = "full"

    # check whether irreducible
    if not isIrreducible(P):
        print("not irreducible")
        x=0


    # P provided is sparse
    zero = np.zeros(n)

    if m_format in ("csr", "csc"):
        # create (P^T - I) with row n-1 set to 1.0s

        pI=[]; pJ=[]; pV = []
        I, J, V = sparse.find(P) 
        for idx in range(0, len(V)):  
            i = I[idx]; j=J[idx]; pr=V[idx]
            if j<n-1:
                pI.append(j); pJ.append(i); pV.append( pr )

        for i in range(0,n-1):
            pI.append(i); pJ.append(i); pV.append(P[i,i]-1.0)

        for j in range(0, n):
            pI.append(n-1); pJ.append(j); pV.append(1.0)      
 
        sP = coo_array((pV,(pI,pJ)), shape=(n,n)).tocsc()

        zero[n-1] = 1.0


        pi = sparse.linalg.spsolve(sP, zero)
        norm = np.sum(pi)
        pi = pi/norm
        for idx in range(len(pi)):
            pi[idx] = max(0.0, pi[idx])
        
        return pi

    for i in range(0,n-1):
        P[i,i] = P[i,i] - 1.0
                
    for j in range(0,n):
        P[n-1,j] = 1.0

    zero[n-1] = 1.0
    pi = linalg.solve(P, zero)
    norm = np.sum(pi)
    pi = pi/norm

    for idx in range(len(pi)):
        pi[idx] = max(0.0, pi[idx])

    return pi

# PI_PtoK computes the vector matrix product pi * P^k .
# The input is a full np two dimensional array.  The method
# builds up the product using the relation P^{a+b} = P^a @ P^b,
# starting with P^1 and at each step doubling the size of the working matrix
# and applying to the accumulating result

def PI_PtoK(pi, P, k, iterate=False):
    result = np.copy(pi)

    if k==0:
        return result

    if not iterate:
        # copy so as not to alter input array
        PtoK = np.copy(P)

        # shift through powers-of-two
        while k>0:
            # a 1 in this 2's position means apply the current power-of-two
            # P product to the accumulating result
            if k & 1 == 1:
                result = result @ PtoK
 
            # double the dimension of P^j
            PtoK = PtoK @ PtoK

            # shift off a lower bit
            k = k >> 1
    else:
        while k>0:
            result = result @ P
            k -= 1

    return result


# spPI_PtoK uses a sparse matrix and repeated row * matrix multiplication
# to build pi(0) @ P^K .  This avoids the cost of matrix-matrix multiplication
# and fill-in that occurs if one builds up explicity P^K

def spPI_PtoK(pi, sP, diag, k):

    # if the format is already csc we're OK
    if sP.format == "csr":
        local_sP = sP.tocsc()
    elif sP.format == "csc":
        local_sP = sP
    else:
        raise ValueError("{sP.format} is unrecognized sparse format of P")

    result = np.copy(pi)
    while k>0:
        result = result @ local_sP
        k -= 1

    return result

def powerMethod(P, tol=1e-5, limit=100000):
    n, _ = P.shape
    u = 1.0/n
    prev = np.asarray([u]*n)
    k=0
    for row in range(0,n):
        rowstring = []
        for col in range(0,n):
            rowstring.append(f"{P[row,col]}")
        print(",".join(rowstring))


    while True:
        present = prev @ P 
        norm = np.sum(present)
        present = present/norm

        if not np.any(np.abs(prev-present) > tol): 
            break
        k += 1
        if k> limit:
            break
        prev = present

    return present, k<limit 

def transientCTMC(Q, pi0, t, qmax=None, tol=1e-12,
                   validation_atol=1e-12, validation_rtol=1e-10):
    """
    Compute pi(t) = pi0 @ exp(Q*t) by uniformization.

    Parameters
    ----------
    Q : (n, n) array
        CTMC generator matrix using the row-vector convention.
    pi0 : (n,) array
        Initial probability distribution (row vector).
    t : float
        Time at which to calculate the distribution.
    qmax : float, optional
        Uniformization rate. Must satisfy qmax >= max(-diag(Q)).
    tol : float
        Upper bound on the omitted Poisson-tail probability.

    validation_atol, validation_rtol : float
        Absolute and relative tolerances used when checking Q and pi0.

    Returns
    -------
    pi_t : (n,) ndarray
        State-occupancy probability vector at time t.
    """

    # turn inputs Q and pi0 into numpy data arrays
    Q = np.asarray(Q, dtype=float)
    pi0 = np.asarray(pi0, dtype=float)

    # error checking on input shapes
    n = Q.shape[0]
    if Q.shape != (n, n):
        raise ValueError("Q must be square.")
    if pi0.shape != (n,):
        raise ValueError("pi0 must have length equal to Q.shape[0].")
    if t < 0:
        raise ValueError("t must be nonnegative.")

    # ensure that pi0 is a probability vector.
    
    non_negative = np.all(pi0 >= 0.0)
    less_than_one = np.all(pi0 <= 1.0)
    sum_is_one = np.isclose(np.sum(pi0), 1.0, atol=tol)
    if not( non_negative and less_than_one and sum_is_one):
        raise ValueError("pi0 must be probability function")
 
    # ensure that Q is proper transition rate matrix, meaning
    # off diagonal entries are non-negative and diagonal entries
    # are negative sum of other elements of the row

    # create a matrix from Q whose diagonal elements are 0.0
    off_diagonal = Q.copy()
    np.fill_diagonal(off_diagonal, 0.0)

    # collect description of any off diagonal element with a negative
    # value, and if any exist, report the first one encountered and die 
    bad_rates = np.argwhere(off_diagonal < -validation_atol)
    if bad_rates.size:
        i, j = bad_rates[0]
        raise ValueError(
            "Q has a negative off-diagonal transition rate: "
            f"Q[{i}, {j}] = {Q[i, j]}."
        )

    # Check Q[i,i] = -sum_{j != i} Q[i,j].
    # Create a vector of the row sums of the matrix zero'd diagonal 
    expected_diagonal = -off_diagonal.sum(axis=1)
    actual_diagonal = np.diag(Q)

    bad_rows = np.flatnonzero(
        # isclose() compares two equal sized arrays for elements
        # that are 'close' to each other. a and b are deemed to be close
        # if absolute(a-b) <= (atol + rtol*absolute(b). Returns
        # vector of Booleans indicating which components are close
        # so ~ inverts the vector so that True means the components are not close
        ~np.isclose(
            actual_diagonal,
            expected_diagonal,
            atol=validation_atol,
            rtol=validation_rtol,
        )
    )

    # bad_rows non-empty means there is a problem
    if bad_rows.size:
        i = bad_rows[0]
        raise ValueError(
            f"Invalid generator row {i}: diagonal entry is "
            f"{actual_diagonal[i]}, but the negative sum of the "
            f"off-diagonal entries is {expected_diagonal[i]}."
        )

    # the uniformation rate has to be at least as large in magnitude
    # as the maximum diagonal element of Q
    minimum_qmax = np.max(-np.diag(Q))
    if qmax is None:
        qmax = minimum_qmax
    if qmax < minimum_qmax:
        raise ValueError(
            f"qmax must be at least {minimum_qmax}."
        )

    # sometimes do nothing
    if t == 0 or qmax == 0:
        return pi0.copy()

    # the 'eye' function returns an identity matrix
    P = np.eye(n) + Q / qmax

    # the mean of the Poisson number of uniformized transitions in [0,t]
    lam = qmax * t

    # Retain enough Poisson terms that the omitted upper tail <= tol.
    # Use ppf function (index of CDF F(n) such that 1-F(n) < 1.0-tol)
    # to determine how many k to condition on
    k_max = int(poisson.ppf(1.0 - tol, lam))

    # k = 0 term

    # Recall that Pr{N_t = k} = \frac{ exp(-lam) lam^k }{k!}
    #                         = \frac{ lam }{k} * P{N_t = k-1)
    state = pi0.copy()
    weight = np.exp(-lam)
    result = weight * state

    # Subsequent terms.  
    # Accumulate overall result in array 'result'
    # Use recursive definition of Poisson pmf
    for k in range(1, k_max + 1):
        state = state @ P
        weight *= lam / k
        result += weight * state

    return result


def sparseTransientCTMC(Q, pi0, t, pi_tol=1e-8, tol=1e-12,
                   validation_atol=1e-12, validation_rtol=1e-10):
    """
    Compute pi(t) = pi0 @ exp(Q*t) by uniformization when Q is sparse

    Parameters
    ----------
    Q : sparse representation of (n, n) array
        CTMC generator matrix using the row-vector convention.
    pi0 : (n,) array
        Initial probability distribution (row vector).
    t : float
        Time at which to calculate the distribution.
    pi_tol : float
        Poisson values below pi_tol are taken to be zero, enabling meaningful
        uniformationization calculation 
    tol : float
        Upper bound on the omitted Poisson-tail probability.

    validation_atol, validation_rtol : float
        Absolute and relative tolerances used when checking Q and pi0.

    Returns
    -------
    pi_t : (n,) ndarray
        State-occupancy probability vector at time t.
    """

    # turn pi0 into numpy data array
    pi0 = np.asarray(pi0, dtype=float)

    # error checking on input shapes
    n = Q.shape[0]
    if Q.shape != (n, n):
        raise ValueError("Q must be square.")
    if pi0.shape != (n,):
        raise ValueError("pi0 must have length equal to Q.shape[0].")
    if t < 0:
        raise ValueError("t must be nonnegative.")

    # ensure that pi0 is a probability vector.
    
    non_negative = np.all(pi0 >= 0.0)
    less_than_one = np.all(pi0 <= 1.0)
    sum_is_one = np.isclose(np.sum(pi0), 1.0, atol=tol)
    if not( non_negative and less_than_one and sum_is_one):
        raise ValueError("pi0 must be probability function")

    diag = np.zeros(n)
    # pull diagonal elements off Q
    for row in range(0, n):
        diag[row] = Q[row, row]
 
    # ensure that Q is proper transition rate matrix, meaning
    # off diagonal entries are non-negative and diagonal entries
    # are negative sum of other elements of the row
    I, J, vQ = sparse.find(Q)
    row_sum = np.zeros(n)
    for idx in range(0, len(vQ)):
        i = I[idx]; j = J[idx]; q = vQ[idx]
        if i==j:
            continue
        if q < -validation_atol:
            raise ValueError(
                "Q has a negative off-diagonal transition rate: "
                f"Q[{i}, {j}] = {q}."
            )
        row_sum[i] += q 

    # check that diagonal is negative of row sum
    for row in range(0, n):
        if not np.isclose(-row_sum[row], diag[row]):
            raise ValueError(
                "diagonal for state {i} is not sum of rates of row"
            )
    
    # the uniformation rate has to be at least as large in magnitude
    # as the maximum diagonal element of Q
    qmax = np.max(-diag)

    # sometimes do nothing
    if t == 0 or qmax == 0:
        return pi0.copy()

    # create a sparse representation of 
    #    np.eye(n) + Q / qmax
    uQ = vQ/qmax
    for idx in range(0, len(vQ)):
        if I[idx] == J[idx]:
            uQ[idx] += 1.0
            diag[ I[idx] ] = uQ[idx]

    P = coo_array((uQ,(I,J)), shape=(n,n)).tocsc()

    # the mean of the Poisson number of uniformized transitions in [0,t]
    lam = qmax * t

    # We're going to ignore numbers of arrivals whose probabilities are too small.
    # Start by finding the smallest k for which the probability of the Poisson 
    # being less than k is at least as great as pi_tol
    k_cdf = int(poisson.ppf(pi_tol, lam))

    # so the smallest k_min for which the probability of k arrivals is
    # greater than pi_tol is no greater than k_cdf, so we can do
    # a binary search to find k_min
    kLow = 0; kHigh = k_cdf
    while kHigh-kLow > 2:
        kMid = int((kHigh+kLow)/2)
        if pi_tol < poisson.pmf(k=kMid, mu=lam):
            kHigh = kMid
        else:
            kLow = kMid+1
       
    k_min = kHigh

    # Retain enough Poisson terms that the omitted upper tail <= tol.
    # Use ppf function (index of CDF F(n) such that 1-F(n) < 1.0-tol)
    # to determine how many k to condition on
    k_max = int(poisson.ppf(1.0 - tol, lam))
    state = np.asarray(pi0, dtype=float)

    # bring state up to k_min
    for k in range(0, k_min):
        state = state @ P

    # construct a weight based on the assumption that there are no arrivals 
    # less than k_min, and so the weight to apply to the first term in the sum
    # is the probability mass value at k_min
    #
    weight = poisson.pmf(k=k_min, mu=lam)

    # initialize result
    result = weight * state

    # Subsequent terms.  
    # Accumulate overall result in array 'result'
    # Use recursive definition of Poisson pmf
    for k in range(k_min+1, k_max + 1):
        state = state @ P
        weight *= lam / k
        result += weight * state

    return result

def steadyStateCTMC(Q):
    # create pQ
    n, _ = Q.shape
    try:
        m_format = Q.format
    except:
        m_format = "full"

    # Q provided is sparse
    diag = np.zeros(n) 
    zero = np.zeros(n)
    m    = np.zeros(n)

    if m_format in ("csr", "csc"):
        pI=[]; pJ=[]; pV = []
        I, J, V = sparse.find(Q) 
        for idx in range(0, len(V)):  
            i = I[idx]; j=J[idx]; qr = V[idx]
            if i != j:
                pI.append(i); pJ.append(j)
                pV.append( -qr/Q[i,i])

        for i in range(0,n):
            m[i] = -1.0/Q[i,i]

        sP = coo_array((pV,(pI,pJ)), shape=(n,n)).tocsc()
        alpha = steadyStateDTMC(sP)
    else:
        P = np.zeros((n,n))
        for i in range(0,n):
            for j in range(0,n):
                if i != j:
                    P[i,j] = -Q[i,j]/Q[i,i]
                    
        alpha = steadyStateDTMC(P)

    m_alpha_prod = alpha * m
    norm = np.sum(m_alpha_prod)
    pi = m_alpha_prod/norm
    return pi

