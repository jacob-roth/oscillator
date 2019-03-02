
import numpy as np
from scipy import *
from pylab import *

T= 0.15
T= 1.0
T= 0.25

def V(x):
    return k*(1-x**2)*2

def f(x):
# fill in as in nonlinear_hints.py:
    # calculate -dV/dx: ie "negative force"
    return 4*k*x*(1-x**2)

Seed = 10000
seed(Seed)
dt = 0.01
k = 1.0
beta = 1.0/T
gamma = 2.0
amplitude = sqrt(2.0*dt*gamma*T)
threshold = 0.0


def escape_from_hole(x0=-1, v=0.1, threshold = 0.0):
    x = x0
    t=0
    while True:
        #x #+= dt*???fill in as in nonlinear_hints.py
        # x += dt*(f(x) + amplitude*standard_normal())
        x += v * dt
        v += dt*(-gamma*v + f(x)) + amplitude*standard_normal()
        t += dt
        if x > threshold:
            return t

t_list = []

NumAve = 100
for i in range(NumAve):
    t_list.append(escape_from_hole())
    print "number completed = ", i+1


n, bins, patches = hist(t_list, 10, normed=1)
print "simulated escape time = ", mean(t_list)
print "analytic escape time  = ", 1/np.exp(- (V(threshold) - V(-1)) / T)

print "simulated escape rate = ", 1/mean(t_list)
print "analytic escape rate  = ", np.exp(- (V(threshold) - V(-1)) / T) / (2.0 * np.pi * gamma)
show()
