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

#Example

std = df.Sales.std()        
mu = df.Sales.mean()      
skew_actual =  df.Sales.skew()      
print(round(skew_actual,2))

skew_gamma = 2*std/mu #Gamma
print(round(skew_gamma,2))

d_min = 400
skew_gamma_p = 2*std/(mu-d_min) #Gamma and d_min
print(round(skew_gamma_p,2))
