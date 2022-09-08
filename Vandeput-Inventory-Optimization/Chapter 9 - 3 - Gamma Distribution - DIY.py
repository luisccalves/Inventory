import numpy as np
import pandas as pd
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt

def get_data(car_maker):
    df = pd.read_csv("norway_new_car_sales_by_make.csv")
    df["Date"] = pd.to_datetime(df["Year"].astype(str)+df["Month"].astype(str),format="%Y%m")
    df = (df.loc[df["Make"] == car_maker,["Date","Quantity"]]
        .rename(columns={"Quantity":"Sales"}).set_index("Date"))
    return df   
df = get_data("Ford")

##########################

#DIY 1

std = df.Sales.std()        
mu = df.Sales.mean()        
shape = mu**2/std**2        #k
scale = std**2/mu           #theta

x_min = gamma.ppf(0.01, shape, scale=scale)
x_max = gamma.ppf(0.99, shape, scale=scale)
x = np.linspace(x_min,x_max, 200)
y = gamma.pdf(x, shape, scale=scale)
plt.plot(x,y)

########################

#DIY 2

std = df.Sales.std()        #std = std'
d_min = df.Sales.min()      #min_d
mu = df.Sales.mean()        #mu
mu_p = mu - d_min           #mu'
shape_p = mu_p**2/std**2    #k'
scale_p = std**2/mu_p       #theta'

x_min = gamma.ppf(0.01, shape_p, loc=d_min, scale=scale_p)
x_max = gamma.ppf(0.99, shape_p, loc=d_min, scale=scale_p)
x = np.linspace(x_min,x_max, 200)
y = gamma.pdf(x, shape_p, loc=d_min,scale=scale_p)
plt.plot(x,y)

shape_auto, min_auto, scale_auto = gamma.fit(df.Sales)

########################

#DIY 3

def rmse_percent(a,b):
    rmse = np.sqrt(sum((a-b)**2)/len(a))/np.mean(b)
    return round(rmse*100,1)
    
hist_range = df.Sales.min()*0.8,df.Sales.max()*1.2
y_actuals, edges = np.histogram(df, bins=30, density=True, range=hist_range)
x = (edges + np.roll(edges, -1))[:-1] / 2.0

mu = df.Sales.mean()
std = df.Sales.std()
y_normal = norm.pdf(x, mu, std)

shape = mu**2/std**2
scale = std**2/mu
y_gamma = gamma.pdf(x, shape, loc=0, scale=scale)

d_min = df.Sales.min()
mu_p = mu - d_min
shape_p = mu_p**2/std**2
scale_p = std**2/mu_p
y_gamma_p = gamma.pdf(x, shape_p, loc=d_min, scale=scale_p)

shape_auto, min_auto, scale_auto = gamma.fit(df.Sales)
y_gamma_auto = gamma.pdf(x, shape_auto, loc=min_auto, scale=scale_auto)

y_actuals /= y_actuals.sum()
y_normal /= y_normal.sum()
y_gamma /= y_gamma.sum()
y_gamma_p /= y_gamma_p.sum()
y_gamma_auto /= y_gamma_auto.sum()

rmse_normal = rmse_percent(y_actuals,y_normal)
rmse_gamma = rmse_percent(y_actuals,y_gamma)
rmse_gamma_p = rmse_percent(y_actuals,y_gamma_p)
rmse_gamma_auto = rmse_percent(y_actuals,y_gamma_auto)

print(rmse_normal)
print(rmse_gamma)
print(rmse_gamma_p)
print(rmse_gamma_auto)

#y_kde = stats.gaussian_kde(df.values.reshape(1,-1)).pdf(x)
#y_kde /= y_kde.sum()
#rmse_kde = rmse(y_actuals,y_kde)
#print(rmse_kde.round(4))
