import numpy as np
from scipy.stats import gamma, norm
from scipy import optimize

#################################################

#DIY #1 Loss 

def gamma_loss(inv,mu,std):
    shape = mu**2/std**2 #k
    scale = std**2/mu    #theta
    loss = shape*scale*(1-gamma.cdf(inv, shape+1, scale=scale))  - inv*(1-gamma.cdf(inv,shape,scale=scale))   
    return loss

def normal_loss(inv,mu,std):
    return std**2*norm.pdf(inv, mu, std) + (mu - inv)*(1-norm.cdf(inv, mu, std))

inv, mu, std = 120, 100, 50, 
loss = gamma_loss(inv,mu,std)
loss2 = normal_loss(inv,mu,std)
print(loss)
print(loss2)

##################################################

#DIY #2 Inverse Loss 

def gamma_loss_inverse(x_mu, x_std ,d_c, beta): 
    target = d_c*(1-beta)
    shape = x_mu**2/x_std**2 
    scale = x_std**2/x_mu

    def unit_shorts(inv):
        return shape*scale*(1-gamma.cdf(inv, shape+1, scale=scale)) - inv*(1-gamma.cdf(inv, shape, scale=scale))  
    
    def f(inv):
        return abs(unit_shorts(inv) - target)
    
    result = optimize.minimize_scalar(f,bounds=(x_mu, x_mu+x_std*5), method='bounded')
    return result.x

beta = 0.95
R, L = 1, 4
d_mu, d_std = 800, 200

d_c = R*d_mu
x_mu = (R+L)*d_mu
x_std = np.sqrt(R+L)*d_std

S = round(gamma_loss_inverse(x_mu, x_std, d_c, beta))
fill_rate = 1-gamma_loss(S, x_mu, x_std)/d_c
print("S:",int(S),"\tFill Rate:",round(fill_rate,3)*100)

beta = 0.95
R, L = 1, 4
d_mu, d_std, d_min = 800, 200, 400

d_c = R*d_mu
x_mu_p = (R+L)*(d_mu - d_min)
x_std = np.sqrt(R+L)*d_std

S_p = round(gamma_loss_inverse(x_mu_p, x_std, d_c, beta)) 
fill_rate = 1-gamma_loss(S_p, x_mu_p, x_std)/d_c
S = S_p+ d_min*(R+L)
print("S:",int(S),"\tFill Rate:",round(fill_rate,3)*100)

