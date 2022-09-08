import numpy as np
from scipy.stats import norm

def normal_loss(inv,mu,std):
    return std**2*norm.pdf(inv, mu, std) + (mu - inv)*(1-norm.cdf(inv, mu, std))

def normal_loss_standard(x):
    return norm.pdf(x) - x*(1-norm.cdf(x))

def f(x):
    return abs(normal_loss_standard(x) - target)

d_c, x_std, beta = 100, 50, 0.99
target = d_c*(1-beta)/x_std

from scipy import optimize
result = optimize.minimize_scalar(f)
print(result)
z = result.x
print(z)

coefficients = [ 4.41738119e-09, 1.79200966e-07, 3.01634229e-06,
2.63537452e-05, 1.12381749e-04, 5.71289020e-06,
-2.64198510e-03, -1.59986142e-02, -5.60399292e-02,
-1.48968884e-01, -3.68776346e-01, -1.22551895e+00,
-8.99375602e-01]

def inverse_standard_loss(target):
    x = np.log(target)
    z = np.polyval(coefficients, x)
    return z

z2 = inverse_standard_loss(target)
print(z2)

d_c, x_std, beta = 250, 30, 0.98
target = d_c*(1-beta)/x_std
z = inverse_standard_loss(target)
print(z)
print(norm.cdf(z))