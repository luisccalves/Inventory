import numpy as np
from scipy.stats import norm

coefficients = [ 4.41738119e-09, 1.79200966e-07, 3.01634229e-06,
2.63537452e-05, 1.12381749e-04, 5.71289020e-06,
-2.64198510e-03, -1.59986142e-02, -5.60399292e-02,
-1.48968884e-01, -3.68776346e-01, -1.22551895e+00,
-8.99375602e-01]

def inverse_standard_loss(target):
    x = np.log(target)
    z = np.polyval(coefficients, x)
    return z

time = 100000
d_mu = 100
d_std = 25
d = np.maximum(np.random.normal(d_mu,d_std,time).round(0).astype(int),0)

L, L_std = 1, 0
R = 4
x_std = np.sqrt((L+R)*d_std**2+L_std**2*d_mu**2)

beta = 0.9
d_c = d_mu * R
target = d_c*(1-beta)/x_std
z = inverse_standard_loss(target)
alpha = round(norm.cdf(z),3)

Ss = np.round(x_std*z).astype(int) 
#Ss = Ss - d_c*(1-beta) #Uncomment if excess demand result in lost sales rather than backorders
Cs = 1/2 * d_mu * R
Is = d_mu * L
S = Ss + 2*Cs + Is

hand = np.zeros(time,dtype=int)
transit = np.zeros((time,L+7*L_std+1),dtype=int)

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
        actual_L = int(round(max(np.random.normal(L,L_std),0),0))
        net = hand[t] + transit[t].sum()    
        transit[t,actual_L] = S - net
        
print("Alpha:",round(alpha*100,1))
print("Beta:",beta*100)

fill_rate = 1-unit_shorts.sum()/sum(d)
print("Fill Rate:",round(fill_rate*100,1))

SL_alpha = 1-sum(stockout_cycle)/len(stockout_cycle)
print("Cycle Service Level:",round(SL_alpha*100,1))

SL_period = 1-sum(stockout_period)/time
print("Period Service Level:",round(SL_period*100,1))