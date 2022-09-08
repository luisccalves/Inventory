from scipy.stats import norm

def normal_loss(inv,mu,std):
    return std**2*norm.pdf(inv, mu, std) + (mu - inv)*(1-norm.cdf(inv, mu, std))

def normal_loss_standard(x):
    return norm.pdf(x) - x*(1-norm.cdf(x))

inv, x_mu, x_std = 120, 100, 50
print(normal_loss(inv,x_mu,x_std))

print(x_std*normal_loss_standard((inv-x_mu)/x_std))

inv, x_mu, x_std  = 270, 250, 30
lost = x_std*normal_loss_standard((inv-x_mu)/x_std)
print(round(lost,2))
beta = 1-lost/x_mu
print(round(beta,2))
z = (inv - x_mu)/x_std
alpha = norm.cdf(z)
print(round(alpha,2))