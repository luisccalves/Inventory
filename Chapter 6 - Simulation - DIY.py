import numpy as np
from scipy.stats import norm

time = 100000
d_mu = 100
d_std = 25
d = np.maximum(np.random.normal(d_mu,d_std,time).round(0).astype(int),0)

L, L_std = 2, 0
R = 4
alpha = 0.95

z = norm.ppf(alpha)
x_std = np.sqrt((L+R)*d_std**2+L_std**2*d_mu**2)
Ss = np.round(x_std*z).astype(int)
Cs = 1/2 * d_mu * R
Is = d_mu * L
S = Ss + 2*Cs + Is

hand = np.zeros(time,dtype=int)
transit = np.zeros((time,L+5*L_std+1),dtype=int)

stockout_period = np.full(time,False,dtype=bool)
stockout_cycle = []

hand[0] = S - d[0]
transit[0,L] = d[0]

for t in range(1,time):
    if transit[t-1,0]>0:
        stockout_cycle.append(stockout_period[t-1])
    hand[t] = hand[t-1] - d[t] + transit[t-1,0]
    stockout_period[t] = hand[t] < 0
#    hand[t] = max(0,hand[t]) #Uncomment if excess demand result in lost sales rather than backorders
    transit[t,:-1] = transit[t-1,1:]        
    if t%R==0:
        actual_L = int(round(max(np.random.normal(L,L_std),0),0))
#        try:
#            max_L = int(max(np.argwhere(transit[t]>0)))
#        except:
#            max_L = 0
#        actual_L = max(actual_L,max_L)
        net = hand[t] + transit[t].sum()    
        transit[t,actual_L] = S - net

print("Alpha:",alpha*100)
SL_alpha = 1-sum(stockout_cycle)/len(stockout_cycle)
print("Cycle Service Level:",round(SL_alpha*100,1))

SL_period = 1-sum(stockout_period)/time
print("Period Service Level:",round(SL_period*100,1))