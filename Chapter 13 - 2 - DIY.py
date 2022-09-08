import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.stats import norm

# STEP #1 - Demand data

def get_data(car_maker):
    df = pd.read_csv("/norway_new_car_sales_by_make.csv")
    df["Date"] = pd.to_datetime(df["Year"].astype(str)+df["Month"].astype(str),format="%Y%m")
    df = (df.loc[df["Make"] == car_maker,["Date","Quantity"]]
        .rename(columns={"Quantity":"Sales"}).set_index("Date"))
    return df     

df = get_data("Toyota")
df = df.iloc[1:] #Remove one outlier

# STEP #2 - Demand distribution estimation

# Create the demand pmf
bandwidth = df["Sales"].std() / (df["Sales"].count()**(1/5))
lower = np.floor(df["Sales"].min() - 3 * bandwidth)
upper = np.ceil(df["Sales"].max() + 3 * bandwidth)
x = np.arange(lower,upper)
kde_dist = gaussian_kde(df["Sales"],bw_method="scott")
pmf = kde_dist.pdf(x)
pmf = pmf/sum(pmf)

# We set the demand once for all simulation. We also set the random seed.
np.random.seed(0)

# Attribute function
def attributes(pmf,x):
    mu = sum(pmf*x)
    std = np.sqrt(sum(x**2*pmf) - sum(pmf*x)**2)
    return mu, std

# Compute demand attributes
d_mu, d_std = attributes(pmf,x)
time = 20000  

d = np.random.choice(x, size=time, p=pmf)

# STEP 3 - Cost function

L_x = np.array([3,4,5])
L_pmf = np.array([0.1,0.7,0.2])
L_mu, L_std = attributes(L_pmf,L_x)
L_median = 4
L_max = 5
k = 2000  #Fixed cost per transaction
h = 1.25  #Holding cost per unit per period
b = 25  #Backlog cost per unit PER PERIOD

time = 20000
R = 1

def simulation_RsQ(x):
    
    hand = np.zeros(time,dtype=int)
    transit = np.zeros((time,L_max),dtype=int)
    
    s = int(round(x[0]))
    Q = int(round(x[1]))
    
    hand[0] = s + Q - d[0]
    transit[1,L_median] = d[0]
      
    p = np.zeros(time) 
    p[0] = s + Q - d[0]/2
    c_k = k 
    c_h = h*p[0] 
    c_b = 0 
    
    for t in range(1,time):    
        hand[t] = hand[t-1] - d[t] + transit[t-1,0]
        if t < time-1:
            transit[t+1,:-1] = transit[t,1:]        
            if t%R==0:
                net = hand[t] + transit[t].sum()    
                if net <= s:
                    actual_L = np.random.choice(L_x, 1, p=L_pmf)
                    transit[t+1,actual_L-1] = (1+ (s-net)//Q)*Q         
                    c_k += k  
        if hand[t] > 0: 
            p[t] = (hand[t-1] + transit[t-1,0] + hand[t])/2
        else: 
            p[t] = max(hand[t-1] + transit[t-1,0],0)**2/max(d[t],1)/2   
        c_h += h*p[t]
        c_b += b*max(0,-hand[t]) 
    
    cost = (c_h+c_b+c_k)/time
    print("\ts, Q, Cost:",s,Q,round(cost,0).astype(int))
    
    return cost

Q = np.sqrt(2*k*d_mu/h)
print("EOQ\t\t",int(round(Q)))
Q = np.sqrt(2*(k+Q*0.02*b*1.1)*d_mu/h)
print("EOQ stochastic\t",int(round(Q)))
alpha_opt1 = 1 - h*R/(b*1.1)
alpha_opt2 = 1 - h*Q/(b*1.1*d_mu)
alpha_opt = (alpha_opt1+alpha_opt2)/2
x_std = np.sqrt((L_mu+R+1/2)*d_std**2 + L_std**2*d_mu**2)
Ss = x_std*norm.ppf(alpha_opt)
s = Ss + (L_mu+1/2)*d_mu



import scipy.optimize
print("Nelder-Mead")
res = scipy.optimize.minimize(fun=simulation_RsQ, x0=np.array([s,Q]), 
                               method='Nelder-Mead',options={'maxiter':50})
print("values:",res.x)
print("results:", res.fun)