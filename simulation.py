

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

time = 200
d_mu = 33.80
d_std = 52.14
# d = np.maximum(np.random.normal(d_mu, d_std, time).round(0).astype(int), 0)
d= abs(np.random.normal(d_mu, d_std, time))
print(d)

L, R, alpha = 4, 1, 0.95
z = norm.ppf(alpha)
x_std = np.sqrt(L+R)*d_std
print(z)
print(x_std)

Ss = np.round(x_std*z).astype(int)
Cs = 1/2 * d_mu * R
Is = d_mu * L
S = Ss + 2*Cs + Is

hand = np.zeros(time, dtype=int)
transit = np.zeros((time, L+1), dtype=int)

stock_out_period = np.full(time, False, dtype=bool)
stock_out_cycle = []

hand[0] = S - d[0]
transit[1, -1] = d[0]

for t in range(1, time):
    if transit[t-1, 0] > 0:
        stock_out_cycle.append(stock_out_period[t-1])
    hand[t] = hand[t-1] - d[t] + transit[t-1, 0]
    stock_out_period[t] = hand[t] < 0
    transit[t, :-1] = transit[t-1, 1:]
    if 0 == t % R:
        net = hand[t] + transit[t].sum()
        transit[t, L] = S - net

df = pd.DataFrame(data={'Demand': d, 'On-hand': hand,
                  'In-transit': list(transit)})
df = df.iloc[R+L:, :]  # Remove initialization periods
print(df)
df['On-hand'].plot(title='Inventory Policy (%d,%d)' %
                             (R, S), ylim=(0, S), legend=True)
plt.savefig('grafico.png')

print('Alpha:', alpha*100)
SL_alpha = 1-sum(stock_out_cycle)/len(stock_out_cycle)
print('Cycle Service Level:', round(SL_alpha*100, 1))
SL_period = 1-sum(stock_out_period)/time
print('Period Service Level:', round(SL_period*100, 1))
