
import numpy
from scipy import *
from pylab import *
import time
from visual import *
import sys


#Refer to the code in problem 2 in this homework, harmonic.py for an
# more detailed explanation of how this works. It is quite similar
def V(x):
    return k*(1-x**2)**2
def f(x):
# calculate -dV/dx:
    return 4*k*x*(1-x**2)

Seed = 10000
seed(Seed)
#time step
dt = 0.01
#constant used in the potential (no longer the spring constant)
k = 1.0
#Temperature:
#T = 0.007
#T= 0.01
#T= 0.05
#T= 0.5
T= 0.1
beta = 1.0/T
T_ =sqrt(2.0*T/dt)
x= 0.0
x0=0
graphics = 1
a = []


if (graphics):
   spring = helix(pos=(0,0,x0), axis=(1,0,0), radius=0.5)
   floor = box (pos=(0,-1,-1), length=10, height=0.5, width=1, color=color.blue)
   ball = sphere (pos=(0,0,0), radius=0.3, color=color.red)

count = 0
time = 50000
for t in arange(0,time,dt):
   # same as harmonic_hints.py but with -k*x replaced with f(x)
   x += dt*(f(x) + T_*standard_normal())
   a.append(x)

   if (graphics and count%10000 == 0):
      #rate(10)
      ball.pos = (x,0,0)
      spring.pos = (x0,0,0)
      spring.axis=(x+x0,0,0)
	    
   count += 1

figure(3)
n, bins, patches = hist(a, 100, normed=1)

xmin = -2.0
xmax = 2.0
x = arange(xmin,xmax,bins[1]-bins[0])
P = exp(-beta*V(x))

norm = sum(P)*(bins[1]-bins[0])
    
plot(x,P/norm)
xlim([xmin,xmax])
figure(1)
plot(arange(0,time,dt),a)
   
show()


