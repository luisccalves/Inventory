import numpy as np
from scipy.stats import skew
data = np.array([10, 5, 14, 25, 4, 3, 6, 5])
print(skew(data))

m = np.mean(data)
s = np.std(data)
print(sum(((data - m)/s)**3)/len(data))