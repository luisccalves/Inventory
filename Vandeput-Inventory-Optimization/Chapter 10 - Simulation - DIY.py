import numpy as np
from scipy.stats import norm

time = 100000
d_mu = 100
d_std = 25
d = np.maximum(np.random.normal(d_mu,d_std,time).round(0).astype(int),0)

alpha = 0.95
z = norm.ppf(alpha)

sites = 3
L = np.array([4,3,2])

#Compute safety stocks needed at each node
#x_tau = np.array([4,3,3]) #if case 1
#x_tau = np.array([0,7,3]) #if case 2
#x_tau = np.array([4,0,6]) #if case 3
x_tau = np.array([0,0,10]) #if case 4
print("Risk-periods\t",x_tau)
x_std = np.sqrt(x_tau)*d_std
Ss = np.round(x_std*z).astype(int)

# Order-up-to level (S) per node
#We compute it based on the risk period of each site
S = Ss + x_tau*d_mu
# Order-up-to level per echelon
S_echelon = (np.cumsum(S[::-1])[::-1]).astype(int)

print("Echelon policy\t",S_echelon)

hand = np.zeros([sites,time],dtype=int)
transit = np.zeros([sites,time,max(L)+1],dtype=int)

#Initiliaze timestep #0
hand[:,0] = S

stockout_period = np.full(time,False,dtype=bool)
unit_shorts = np.zeros(time,dtype=int)

for t in range(1,time):
#    print()
#    print("time\t",t)
    
    # Arrival of orders
    hand[:,t] = hand[:,t-1] + transit[:,t-1,0]
#    print("Reception")
#    print(transit[:,t-1,0])
    
    # Transit move by one step
    transit[:,t,:-1] = transit[:,t-1,1:]  
            
    # Demand consumption
    unit_shorts[t] = max(0,d[t] - max(0,hand[-1,t]))
    hand[-1,t] = hand[-1,t] - d[t]

    # Check for service level
    stockout_period[t] = hand[-1,t] < 0
#    hand[t] = max(0,hand[t]) #Uncomment if excess demand result in lost sales rather than backorders

#    print()      
#    print("On-hand inventory before orders")
#    print(hand[:,t])
#    print("Transit inventory")
#    print(transit[:,t])
#    
    # New orders
    for site in range(sites):
        #Compute the net inventory for the echelon (i.e. the inventory of this node plus those downstream)
        net = hand[site:,t].sum() + transit[site:,t].sum()
        order = S_echelon[site] - net
        # If this is not a supplying node, we need to check for stock availability to the supplying node
        if site > 0: #all nodes but the supply node
            available = hand[site-1,t] + transit[site-1,t,0] #stock available at the supplying node, it includes the stock that arrives between two timesteps
            order = min(available, order) #constraint the order
            hand[site-1,t] -= order #consume the order at the supplying node
            transit[site,t,L[site]] = order
        elif site == 0: #supply node
            transit[site,t,L[site]] = order

#    print()        
#    print("After orders")
#    print(hand[:,t])
#    print(transit[:,t])
     
print()          
print("Alpha:",alpha*100)

fill_rate = 1-unit_shorts[sum(x_tau):].sum()/sum(d[sum(x_tau):])
print("Fill Rate:",round(fill_rate*100,1))

SL_period = 1-sum(stockout_period)/time
print("Period Service Level:",round(SL_period*100,2))

