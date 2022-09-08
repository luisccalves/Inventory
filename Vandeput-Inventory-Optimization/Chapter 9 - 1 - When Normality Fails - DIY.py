import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
   
def get_data(car_maker):
    df = pd.read_csv("norway_new_car_sales_by_make.csv")
    df["Date"] = pd.to_datetime(df["Year"].astype(str)+df["Month"].astype(str),format="%Y%m")
    df = (df.loc[df["Make"] == car_maker,["Date","Quantity"]]
        .rename(columns={"Quantity":"Sales"}).set_index("Date"))
    return df     

df = get_data("Ford")
df.plot(figsize=(8,4))

y_actuals, edges = np.histogram(df, bins=30, density=True)
print(y_actuals.sum())
y_actuals = y_actuals/sum(y_actuals)
print(y_actuals.sum())

hist_range = (df.Sales.min()*0.8,df.Sales.max()*1.2)
y_actuals, edges = np.histogram(df, bins=30, density=True, range=hist_range)
y_actuals = y_actuals/sum(y_actuals)

plt.hist(df.Sales.values,bins=30,density=True,label="Sales", range=hist_range)
plt.show()

df.plot(kind="hist",density=True,bins=30, range=hist_range)