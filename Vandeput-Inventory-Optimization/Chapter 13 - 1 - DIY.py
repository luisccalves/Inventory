import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.stats import norm

# STEP #1 - Demand data

def get_data(car_maker):
    df = pd.read_csv("norway_new_car_sales_by_make.csv")
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
k = 1000  #Fixed cost per transaction
h = 1.25  #Holding cost per unit per period
b = 25  #Backlog cost per unit per period

def simulation(R,Ss):
        
    S = round(d_mu * (R+L_mu) + round(Ss)).astype(int)
    
    hand = np.zeros(time,dtype=int)
    transit = np.zeros((time,L_max+1),dtype=int)
    unit_shorts = np.zeros(time,dtype=int)
    
    stockout_period = np.full(time,False,dtype=bool)
    stockout_cycle = []
    
    hand[0] = S - d[0]
    transit[1,L_median] = d[0]
      
    p = np.zeros(time) #Physical stock
    p[0] = S - d[0]/2
    c_k = k #Transaction costs
    c_h = h*p[0] #Holding costs
    c_b = 0 #Backlog costs assuming no backlog during the first period

    for t in range(1,time):
        if transit[t-1,0]>0:
            stockout_cycle.append(stockout_period[t-1])
        unit_shorts[t] = max(0,d[t] - max(0,hand[t-1] + transit[t-1,0]))
        hand[t] = hand[t-1] - d[t] + transit[t-1,0]
        stockout_period[t] = hand[t] < 0
        transit[t,:-1] = transit[t-1,1:]        
        if t%R==0:
            actual_L = np.random.choice(L_x, 1, p=L_pmf)
            net = hand[t] + transit[t].sum()    
            transit[t,actual_L] = S - net
            c_k += k             
        if hand[t] > 0:  #there is enough stock by the end of the period
            p[t] = (hand[t-1] + transit[t-1,0] + hand[t])/2
        else:  #there is not
            p[t] = max(hand[t-1] + transit[t-1,0],0)**2/max(d[t],1)/2   
        c_h += h*p[t]
        c_b += b*max(0,-hand[t]) #backlog cost times the total backlog
            
    SL_alpha = 1-sum(stockout_cycle)/len(stockout_cycle)            
    fill_rate = 1-unit_shorts.sum()/sum(d)
    cost = (c_h+c_b+c_k)/time
#    print("\t\tSs, Cost:",Ss,round(cost,0).astype(int))
    
    return cost, SL_alpha, fill_rate, Ss

# STEP 4 - Optimization function

def find_best_Ss(step_size=1, start=0, threshold=1.1):
    results = [] #result will store, for each simulation, a tuple with 4 values: the cost, the cycle service level, the fill rate and the safety stock.
    Ss_opt = start
    results.append(simulation(R,Ss_opt))
    cost_opt = results[-1][0]
    Ss_new = step_size + start
    results.append(simulation(R,Ss_new))
    cost_new = results[-1][0]
    while cost_new < cost_opt*threshold : 
        if cost_new < cost_opt:
            cost_opt = cost_new
            Ss_opt = Ss_new
            print("New Ss_opt:",Ss_opt)
        Ss_new += step_size
        results.append(simulation(R,Ss_new))
        cost_new = results[-1][0]
    print("Best found:",Ss_opt,cost_opt)
    return results

# STEP 5 - First technique

R = 1
#We start at Ss = 0 and we increase as long as cost decreases
step_size = int(max(1,round(d_mu/100)))
results = find_best_Ss(step_size=step_size, start=0)

def print_results(results):
    df = pd.DataFrame(results)
    df.columns = ["Cost","Cycle Service Level","Fill Rate","Safety Stocks"]
    df = df.set_index("Safety Stocks").sort_index()
    df.plot(secondary_y=["Cycle Service Level","Fill Rate"])
print_results(results) 

# STEP 6 - Smarter parameters

alpha_opt = 1 - h*R/(b*1.1)
x_std = np.sqrt((L_mu+R)*d_std**2 + L_std**2*d_mu**2)
Ss = x_std*norm.ppf(alpha_opt)
Ss = int(round(Ss))

print("Increasing")
step_size = int(max(1,round(d_mu/100)))
results_increase = find_best_Ss(step_size=step_size, start=Ss)

print("Decreasing")
step_size = int(max(1,round(d_mu/100)))
results_decrease = find_best_Ss(step_size=-step_size, start=Ss-step_size)
print_results(results_increase+results_decrease)