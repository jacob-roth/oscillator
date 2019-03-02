
from scipy import *
from pylab import *
import time
graphics = 0
if graphics:
    from visual import *
import sys
from scipy.optimize import leastsq

#get rid of the end of the data if its value is too low because it's quite noisy
def make_cutoff(s, r_min = 0.01):
    out = []
    s0 = s[0]
    y_cut  = s[0]*r_min
    for y in s:
        out.append(y)
        if y < y_cut:
            return out #getting too small so leave keeping only this much data

    return out
        


dt = 0.01
k = 1.0
Temperature = 1.0
amplitude = sqrt(2.0*Temperature/dt)
x= 0.0
x0=0

if (graphics):
   spring = helix(pos=(0,0,x0), axis=(1,0,0), radius=0.5)
   floor = box (pos=(0,-1,-1), length=10, height=0.5, width=1, color=color.blue)
   ball = sphere (pos=(0,0,0), radius=0.3, color=color.red)


#Solves the Langevin eqn for a particle with damping and thermal noise attached to a linear spring
def run_spring(t_max, x=0):
    a = []
    for t in arange(0,t_max,dt):
        x += dt*(-k*x + amplitude*standard_normal())
        a.append(x)

        if (graphics):
            rate(10)
            ball.pos = (x,0,0)
            spring.pos = (x0,0,0)
            spring.axis=(x+x0,0,0)

    return a

def compare_to_histogram(a, Graph=True):
    n, bins, patches = hist(a, 100, normed=1)
    sigma= std(a)
    gauss = exp(-bins*bins/(2*sigma**2))/(sqrt(2*pi)*sigma)
    print "sigma = ", sigma
    if Graph:
        xlabel("x")
        ylabel("P(x)")
        plot(bins,gauss)

def compute_correlation(a, Graph=True):
    f = abs(fft(a))**2
    corr = real(ifft(f))/len(f) #Wiener Khinchine theorem
    #now fit:
    t_window = 6
    n_window = t_window/dt
    t_range = arange(0,t_window,dt)
#   This is how to fit a function. The fitting function is called fitf
    fitf = lambda p, t: p[0]*exp(-t/p[1])
#   We now define a function giving the error from the fit from the data:
    errorf = lambda p, omega, data: fitf(p,omega)-data 
    #initial guess of parameters is called p0. I've chosen p[0] to be the amplitude of the function so:
    p0 = [corr[0],1.0]
#   Now we do the fitting:
    p1, success = leastsq(errorf,p0[:],args=(t_range,corr[0:n_window]))
    print "fitting correlation function: p1 = ", p1

#   Graph both the correlation function data and the fit:
    if Graph:
        xlabel("t")
        ylabel("<x(0)x(t)>")
        plot(t_range, corr[0:n_window],t_range, fitf(p1,t_range))
    return corr

def compute_smoothed_power_spect(corr, smooth_time = 300.):
    var = smooth_time**2
    filter = array([exp(-0.5*k**2/var) + exp(-0.5*(k-len(corr))**2/var) for k in range(len(corr))])
    return(dt*real(fft(filter*corr)))

def fit_spect(s, Graph=True):
    small = make_cutoff(s_total)
    s_limit = size(small)
    t_limit = dt*size(s_total)
    om = (2*pi/t_limit)*arange(0,s_limit,1.0,dtype=float)

    #Fit to a Lorentzian distr:
    fitf = lambda p, omega: p[0]/(1+omega**2/p[1]**2) 
    errorf = lambda p, omega, data: fitf(p,omega)-data 
    p0 = [s[0],2.0]
    p1, success = leastsq(errorf,p0[:],args=(om,small))
    print p1

    n_tot = t_max/dt
    dw = 2*pi/n_tot
    dw_bin = dw*smooth_bins

    damp_time =  k/dt
    w0 = 2*pi/damp_time
    print "w0 = ", w0

    if Graph:
        xlabel(r'$\omega$')
        ylabel(r'$|{\hat x}_\omega|^2$')
        plot(om,small,"b^",om,fitf(p1,om),"r-")

Seed = 10000
seed(Seed)
   
t_max= 10000
#Generate the optical trap position data:
a = run_spring(t_max)


figure(0)
#How well does the histogram of positions match a Gaussian:
compare_to_histogram(a)

#You can change the True for False if you don't want this to
#show up when you look at the movie of the ball and spring
#but it's normally fine leaving the movie on. It just goes very slowly when graphics = 1
if True and graphics == 0:

    smooth_bins = 400
    figure(1)
    #Compute the correlation function and see how well it fits an exponential:
    corr = compute_correlation(a, Graph=True)
    figure(2)
    #Compute the power spectrum. This is a bit tricky because it needs to be
    #slightly smooth to eliminate noise.
    s_total = compute_smoothed_power_spect(corr)
    #Fit and compare this to a Lorentzian:
    fit_spect(s_total, Graph=True)


show()


