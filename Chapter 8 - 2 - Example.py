import numpy as np
import pandas as pd
from scipy.stats import norm

############## Example #1

def normal_loss_standard(x):
    return norm.pdf(x) - x*(1-norm.cdf(x))

def cost(h,d_mu,R,z,x_std,k,b):
    return h*(d_mu*R/2+z*x_std)+k/R+b*x_std*normal_loss_standard(z)/R
    
h = 1.25
d_mu = 100
R = 1
L = 1
z = 1.645
d_std = 25
k = 1000
b = 50

x_std = d_std*np.sqrt(R+L)
print(cost(h,d_mu,R,z,x_std,k,b))

############## Example #2

def CSL_optimal(b,h,R):
    return 1-(h*R)/b

alpha = CSL_optimal(b,h,R)
z = norm.ppf(alpha)
print(cost(h,d_mu,R,z,x_std,k,b))

############## Example #3

df = pd.DataFrame(columns=["Review Period", "Inventory Cost", "Cycle Service Level", "Fill Rate"])
for R in [1,2,3,4,5,6,7]:
    x_std = 25*np.sqrt(R+L)
    alpha = CSL_optimal(b,h,R)
    z = norm.ppf(alpha)
    beta = 1 - x_std*normal_loss_standard(z)/R/d_mu
    df = df.append({"Cycle Service Level":alpha, "Fill Rate":beta, "Inventory Cost":cost(h,d_mu,R,z,x_std,k,b), "Review Period":R},ignore_index=True)

print(df)
df.plot(y=["Inventory Cost","Cycle Service Level","Fill Rate"],x="Review Period",secondary_y=["Cycle Service Level","Fill Rate"],figsize=(8,4))