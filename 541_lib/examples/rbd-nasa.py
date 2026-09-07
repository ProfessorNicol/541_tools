from pathlib import Path
import sys
script_dir = Path(__file__).resolve().parent
target_dir = (script_dir/"../").resolve()
sys.path.append(str(target_dir))

from comb import minDist, maxDist, kofNDist
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import math
import pdb

# Define a grid of points where you want to evaluate the new distribution
x = np.linspace(0, 4000, 4000)

cpuDist = stats.expon(scale=1000)
cpuCDF  = cpuDist.cdf(x)

hsnPrimDist = stats.expon(scale=2000)
hsnPrimCDF = hsnPrimDist.cdf(x)

hsnBckupDist = stats.expon(scale=500)
hsnBckupCDF    = hsnBckupDist.cdf(x)

lsnPrimDist = stats.expon(scale=5000)
lsnPrimCDF = lsnPrimDist.cdf(x)

lsnBckupDist = stats.expon(scale=5000)
lsnBckupCDF    = lsnBckupDist.cdf(x)

cpuSysCDF, _ = kofNDist(2,3, cpuCDF,x)
hsnSysCDF, _ = maxDist([hsnPrimCDF, hsnBckupCDF], x)
lsnSysCDF, _ = maxDist([lsnPrimCDF, lsnBckupCDF], x)

sysCDF, _ = minDist([cpuSysCDF, hsnSysCDF, lsnSysCDF], x)

plt.figure(figsize=(8, 5))
plt.plot(x, sysCDF, label='CDF of System Failure Time', color='blue', linewidth=2)

# 3. Add formatting
plt.title('Cumulative Distribution Function (CDF)')
plt.xlabel('Time t')
plt.ylabel('Cumulative Probability (P(X ≤ t))')
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 1.05) # Probabilities stay between 0 and 1
plt.legend()

# 4. Display the plot
plt.savefig("rbd-nasa.png")
plt.show()
