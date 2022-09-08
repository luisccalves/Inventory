import numpy as np
from scipy.stats import gamma

###########################

#Example #1 Cycle Service Level

d_mu, d_std, d_min = 800, 200, 400
alpha, L, R = 0.95, 4, 1
x_min = (L+R)*d_min
x_mu = (L+R)*d_mu
x_mu_p = x_mu - x_min   #mu_x'
x_std = np.sqrt(L+R)*d_std      #std_x = std_x'
x_shape_p = x_mu_p**2/x_std**2  #k_x'
x_scale_p = x_std**2/x_mu_p     #theta_x'
S = round(gamma.ppf(alpha,x_shape_p,scale=x_scale_p),0) + x_min
Ss = S - x_mu
print(S,Ss)

###########################

#Example #2 Fill Rate

def gamma_loss(inv,mu,std):
    shape = mu**2/std**2 #k
    scale = std**2/mu    #theta
    loss = shape*scale*(1-gamma.cdf(inv, shape+1, scale=scale))  - inv*(1-gamma.cdf(inv,shape,scale=scale))   
    return loss

d_mu, d_std, d_min = 800, 200, 400
alpha, L, R = 0.95, 4, 1
x_mu_p = (L+R)*(d_mu - d_min)   #mu_x'
x_std = np.sqrt(L+R)*d_std      #std_x = std_x'
x_shape_p = x_mu_p**2/x_std**2  #k_x'
x_scale_p = x_std**2/x_mu_p     #theta_x'
S_p = round(gamma.ppf(alpha,x_shape_p,scale=x_scale_p),0) #S'
S = S_p + (L+R)*d_min
unit_shorts = gamma_loss(S-d_min*(R+L),x_mu_p,x_std) 
beta = 1-unit_shorts/(d_mu*R)
print(round(beta*100,1))