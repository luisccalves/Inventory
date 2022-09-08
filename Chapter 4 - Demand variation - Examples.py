from scipy.stats import norm

mu = 100
sigma = 25

alpha = [0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99]
inv = norm.ppf(alpha, mu, sigma)

inv = [100,106,113,121,132,141,149,158]
alpha = norm.cdf(inv, mu, sigma)

mu, sigma, z = 100, 25, 125
alpha = norm.cdf(z, mu, sigma)
print(alpha)

mu, sigma, alpha = 100, 25, 0.95
z = norm.ppf(alpha, mu, sigma)
print(z)
