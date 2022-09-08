import numpy as np

demand = [0,2,4,6,8,10]
pmf = [0.4,0.2,0.2,0.1,0.05,0.05]
quantity = [2,4,6,8,10]
p = 6; c = 2; s = 1

results = np.empty(shape=(len(demand),len(quantity)))
for row, d in enumerate(demand):
    for column, q in enumerate(quantity):
        results[row,column] = min(q,d)*p - q*c + max(0,q-d)*s

print(results)   

profits = np.empty(shape=len(quantity))
for column in range(len(quantity)):
    profits[column] = sum(results[:,column] * pmf)

print(profits)

EOQ = quantity[np.argmax(profits)]
print("EOQ:",EOQ,"\nProfits:",max(profits))




