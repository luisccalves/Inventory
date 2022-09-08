import numpy as np
from scipy.stats import norm

def normal_loss_standard(x):
    return norm.pdf(x) - x*(1-norm.cdf(x))

time = 100000
d_mu = 100
d_std = 25
d = np.maximum(np.random.normal(d_mu,d_std,time).round(0).astype(int),0)

L, L_std = 4, 0
R = 4
alpha = 0.80

z = norm.ppf(alpha)
x_std = np.sqrt((L+R)*d_std**2+L_std**2*d_mu**2)
Ss = np.round(x_std*z).astype(int)
Cs = 1/2 * d_mu * R
Is = d_mu * L
S = Ss + 2*Cs + Is

stockout_period = np.full(time,False,dtype=bool)
stockout_cycle = []

hand = np.zeros(time,dtype=int)
transit = np.zeros((time,L+7*L_std+1),dtype=int)
unit_shorts = np.zeros(time,dtype=int)

hand[0] = S - d[0]
transit[0,L] = d[0]

k = 1000  #Fixed cost per transaction
h = 1.25  #Holding cost per unit per period
b = 50  #Backorder cost per unit

p = np.zeros(time)
p[0] = S - d[0]/2
c_k = k #Transaction costs
c_h = h*p[0] #Holding costs
c_b = 0 #Backorder costs assuming no backorders during the first period

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
        c_k += k             
    if hand[t] > 0:  #there is enough stock by the end of the period
        p[t] = (hand[t-1] + transit[t-1,0] + hand[t])/2
    else:  #there is not
        p[t] = max(hand[t-1] + transit[t-1,0],0)**2/max(d[t],1)/2   
    c_h += h*p[t]
    c_b += b*unit_shorts[t]
                                                         
print("Alpha:",round(alpha*100,1))

SL_alpha = 1-sum(stockout_cycle)/len(stockout_cycle)
print("Cycle Service Level:",round(SL_alpha*100,1))

def cost(h,d_mu,R,L,z,x_std,k,b):
    return h*(d_mu*(R/2)+z*x_std)+k/R+b*x_std*normal_loss_standard(z)/R

print("Model:\t\t",round(cost(h,d_mu,R,L,z,x_std,k,b),1))
print("Simulation:\t",round((c_h+c_b+c_k)/time,1))

print(p.mean())
print(Cs+Ss)