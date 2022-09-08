import numpy as np
from scipy.stats import gamma
from scipy import optimize

def gamma_loss(inv,mu,std):
    shape = mu**2/std**2 #k
    scale = std**2/mu    #theta
    loss = shape*scale*(1-gamma.cdf(inv, shape+1, scale=scale)) - inv*(1-gamma.cdf(inv,shape,scale=scale))   
    return loss

def gamma_loss_inverse(x_mu, x_std, d_c, beta): 
    target = d_c*(1-beta)
    shape = x_mu**2/x_std**2 
    scale = x_std**2/x_mu

    def unit_shorts(inv):
        return shape*scale*(1-gamma.cdf(inv, shape+1, scale=scale)) - inv*(1-gamma.cdf(inv,shape,scale=scale))  
    
    def f(inv):
        return abs(unit_shorts(inv) - target)
    
    result = optimize.minimize_scalar(f,bounds=(x_mu,x_mu+x_std*5), method='bounded')
    return result.x

time = 100000
d_mu, d_std, d_min = 200, 25, 100
d_mu_p = d_mu - d_min           #d_mu'
d_shape_p = d_mu_p**2/d_std**2  #k_d'
d_scale_p = d_std**2/d_mu_p     #theta_d'
d = np.maximum(np.random.gamma(d_shape_p,d_scale_p,time).round(0).astype(int)+d_min,0)

L,L_std, R = 1, 0,  4
d_c = R*d_mu
x_std = np.sqrt((L+R)*d_std**2) #x_std' = x_std
x_mu_p = (R+L)*(d_mu-d_min)     #x_mu'
x_shape_p = x_mu_p**2/x_std**2  #k_x'
x_scale_p = x_std**2/x_mu_p     #theta_x'

#Use if alpha is your target. 
alpha = 0.90
S_p =  round(gamma.ppf(alpha,x_shape_p,scale=x_scale_p),0)
beta = 1-gamma_loss(S_p,x_mu_p,x_std)/d_c
S = S_p + (L+R)*d_min

##Use if beta is your target.
#beta = 0.99
#S_p = round(gamma_loss_inverse(x_mu_p, x_std, d_c, beta))
#alpha = gamma.cdf(S_p,x_shape_p,scale=x_scale_p)
#S = S_p + (L+R)*d_min

#Use if gamma lead time
#L_shape = L**2/L_std**2  #k_L'
#L_scale = L_std**2/L     #theta_L'

Cs = 1/2 * d_mu * R
Is = d_mu * L
Ss = S - 2*Cs - Is

hand = np.zeros(time,dtype=int)
transit = np.zeros((time,L+5*L_std+1),dtype=int)

hand[0] = S - d[0]
transit[0,L] = d[0]

stockout_period = np.full(time,False,dtype=bool)
stockout_cycle = []
unit_shorts = np.zeros(time,dtype=int)

for t in range(1,time):
    if transit[t-1,0]>0:
        stockout_cycle.append(stockout_period[t-1])
    unit_shorts[t] = max(0,d[t] - max(0,hand[t-1] + transit[t-1,0]))
    hand[t] = hand[t-1] - d[t] + transit[t-1,0]
    stockout_period[t] = hand[t] < 0
#    hand[t] = max(0,hand[t]) #Uncomment if excess demand result in lost sales rather than backorders
    transit[t,:-1] = transit[t-1,1:]        
    if t%R==0:
#        actual_L = int(round(np.random.gamma(L_shape,L_scale),0))
        actual_L = int(round(max(np.random.normal(L,L_std),0),0))
        net = hand[t] + transit[t].sum()    
        transit[t,actual_L] = S - net
                
print("Alpha:",round(alpha*100,1))
print("Beta:",round(beta*100,1))

fill_rate = 1-unit_shorts.sum()/d.sum()
print("Fill Rate:",round(fill_rate*100,1))

SL_alpha = 1-sum(stockout_cycle)/len(stockout_cycle)
print("Cycle Service Level:",round(SL_alpha*100,1))

SL_period = 1-stockout_period.sum()/time
print("Period Service Level:",round(SL_period*100,1))