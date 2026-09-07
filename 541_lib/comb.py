import numpy as np
import scipy.stats as stats
from scipy.stats import binom
import math

# minDist returns np arrays giving the cdf and pdf
# of the distribution of the minimum of input
# distributions representing random variables all assumed to be independent
# The inputs are in a list 'dists', all cdfs, and the discretization baseline in x

def minDist(dists, x):

    # sf used to create survivor distributions from the cdfs 
    sf = []
    for dist in dists:
        sf.append( 1.0 - dist )

    # min_sf will accumulate the minimum distribution,
    min_sf = sf[0]

    for idx in range(1,len(sf)):
        # component by component multiplication, each representing
        # Pr{min_sf > t}*Pr{sf[idx] > t} for some t
        min_sf = min_sf*sf[idx]

    # get the cdf from the survivor function
    cdf_min = 1.0 - min_sf

    # use np numerics to estimate a numerical pdf
    pdf_min = np.gradient(cdf_min, x)
    return cdf_min, pdf_min

# maxDist returns np arrays giving the cdf and pdf
# of the distribution of the maximum of input
# distributions representing random variables all assumed to be independent
# The inputs are in a list 'dists', all cdfs, and the discretization baseline in x

def maxDist(dists, x):

    # cdf_max will accumulate the cdf of the desired distribution
    cdf_max = dists[0]


    for idx in range(1, len(dists)):
        # component by component multiplication, each representing
        # Pr{cdf_max <= t *Pr{dists[idx] <= t}  for some t
        cdf_max = cdf_max * dists[idx]

    # use np numerics to estimate a numerical pdf
    pdf_max = np.gradient(cdf_max, x)
    return cdf_max, pdf_max 


# compute the cdf and pdf of the time when k or more of N
# independent and identically distributed random variables
# have values less than or equal to t
# Below k and N are the combinatorial parameters, dist is the i.i.d. distribution,
# and x is the discretization baseline

def kofNDist(k, N, dist, x):
    # initialize the cdf with zeros
    kofN_cdf = np.zeros(len(x))

    # iterate over the times (and probabilities) in the distribution
    for idx, p in enumerate(dist):

        # iterate over full range of elements to select to have times that
        # are no greater than p
        for j in range(k, N+1):
            # use scipy's binom function to avoid user-induced problems with scale of floating point
            # numbers involved in this computation
            kofN_cdf[idx] += binom.pmf(j, N, p)

    # use np numerics to estimate a numerical pdf
    kofN_pdf = np.gradient(kofN_cdf, x)
    return kofN_cdf, kofN_pdf


