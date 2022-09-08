
# Section 2

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

def get_data(car_maker):
    df = pd.read_csv("norway_new_car_sales_by_make.csv")
    df["Date"] = pd.to_datetime(df["Year"].astype(str)+df["Month"].astype(str),format="%Y%m")
    df = (df.loc[df["Make"] == car_maker,["Date","Quantity"]]
        .rename(columns={"Quantity":"Sales"}).set_index("Date"))
    return df     

df = get_data("Toyota")
df = df.iloc[1:] #Remove one outlier

df["Sales"].plot(kind="hist",density=True,bins=30)

#bw = 0.9*1/(df["Sales"].count()**(1/5))
#kde_dist = gaussian_kde(df["Sales"],bw_method=bw)

kde_dist = gaussian_kde(df["Sales"],bw_method="scott")

x = np.linspace(df["Sales"].min()*0.9,df["Sales"].max()*1.1,1000)
y_kde = kde_dist.pdf(x)
plt.plot(x, y_kde)
plt.show()
print(kde_dist.integrate_box_1d(-np.inf,1250))

def kde_cdf(kde_dist,x):
    cdf = []
    for value in x:
        cdf.append(kde_dist.integrate_box_1d(-np.inf,value))
    return cdf
plt.plot(x,kde_cdf(kde_dist,x))
plt.show()


#Final figure
import matplotlib.patches as mpatches
fig, ax = plt.subplots()

df = get_data("Toyota").iloc[1:] 
df["Sales"].plot(ax=ax,kind="hist",density=True,bins=30,alpha=0.6)
patch = mpatches.Patch(color='C0', alpha=0.6,label='Sales')

x = np.linspace(df["Sales"].min()*0.9,df["Sales"].max()*1.1,1000)
y_kde = kde_dist.pdf(x)
plot2 = ax.plot(x, y_kde,lw=1.5)
ax.set_xlabel("Monthly sales")

ax1 = ax.twinx()
ax1.set_ylabel("Frequency")
plot3 = ax1.plot(x,kde_cdf(kde_dist,x),ls="--",color="C2",lw=1.5)

ax.legend([patch]+plot2+plot3, ["Sales","KDE pdf","KDE cdf"],loc="upper left")

plt.show()