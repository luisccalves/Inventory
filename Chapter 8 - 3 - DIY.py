import numpy as np
from scipy.stats import norm

def normal_loss_standard(x):
    return norm.pdf(x) - x*(1-norm.cdf(x))

def EOQ(k,D,h):
    return np.sqrt(2*k*D/h)

def CSL_optimal(h,Q,D,b):
    return 1-(h*Q)/(D*b)

def Q_optimal(k,D,h,b,z,x_std):
    return np.sqrt(2*(k+b*x_std*normal_loss_standard(z))*D/h)

#def cost(h,d_mu,R,L,z,x_std,k,b):
#    return h*(d_mu*(Q/2+L)+z*x_std)+k*D/Q + b*x_std*normal_loss_standard(z)*D/Q
    
def sQ_optimal(k,D,h,b,x_std):
    Q = EOQ(k,D,h)
    Q_old = 0
    while Q_old != Q:
        Q_old = Q
        z = norm.ppf(CSL_optimal(h,Q,D,b))
        Q = round(Q_optimal(k,D,h,b,z,x_std)) 
    return z,Q

k = 1000
D = 100*52
h = 1.25*52
b = 50
x_std = 25
print(sQ_optimal(k,D,h,b,x_std))

    