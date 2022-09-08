import numpy as np

def EOQ(k,D,h,p): #h is a percentage
    return np.sqrt(2*k*D/(h*c))

def Cost(k,D,h,Q,c): #h is a percentage
    return k*D/Q + h*c*Q/2 + c*D


#Case #1
c, k, D, h = 1, 100, 2000, 0.15
Q = EOQ(k,D,h,c)
C = Cost(k,D,h,Q,c)
print(round(Q),"\t",round(C))

#Case 2
c = 0.8
Q = max(2000,EOQ(k,D,h,c))
C = Cost(k,D,h,Q,c)
print(round(Q),"\t",round(C))