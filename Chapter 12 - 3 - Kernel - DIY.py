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

# Section 3

bandwidth = df["Sales"].std() / (df["Sales"].count()**(1/5))
lower = np.floor(df["Sales"].min() - 3 * bandwidth)
upper = np.ceil(df["Sales"].max() + 3 * bandwidth)
x = np.arange(lower,upper)
kde_dist = gaussian_kde(df["Sales"],bw_method="scott")
pmf = kde_dist.pdf(x)
pmf = pmf/sum(pmf)

plt.plot(x,pmf)

random_values = np.random.choice(x, size=10000, p=pmf)


#If negative values
zero = x == 0
negative = x < 0
#Update the PMF at 0
pmf[zero] = pmf[zero] + pmf[negative].sum()
#Remove negative values
zero_arg = np.argmax(x==0)
pmf = pmf[zero_arg:]
x = x[zero_arg:].astype(int)
