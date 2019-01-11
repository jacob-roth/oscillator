import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time


## functions
## -----------------------------------------------------------------------------
def V(x, k):
    return k*(1-x**2)**2

def dVdx(x, k):
    return -4*k*x*(1-x**2)

def feuler(x0, v0, nsteps,
           k=None, dt=None, sigma=None, gamma=None, threshold=None, seed=None):
    ## setup
    np.random.seed(seed)
    xnew = np.copy(x0)
    xold = np.copy(x0)
    v = np.copy(v0)
    zeta = np.random.normal(size=nsteps)
    t = 0.0
    cross = 0
    crosstimes = []
    xhist = []

    ## euler
    for i in range(nsteps):
        xnew = xold + v * dt
        v += dt*(-gamma*v - dVdx(xnew, k)) + sigma*zeta[i]
        t += dt
        if np.sign((xnew - threshold)*(xold - threshold)) < 0:
            cross += 1
            crosstimes += [t]
        xold = xnew
        xhist += [xnew]
    time_to_cross = np.diff(np.hstack([0, crosstimes]))
    return cross, time_to_cross, xhist

def feuler_single(x0, v0,
                  k=None, dt=None, sigma=None, gamma=None, threshold=None, seed=None):
    ## setup
    np.random.seed(seed)
    xnew = np.copy(x0)
    xold = np.copy(x0)
    v = np.copy(v0)
    t = 0.0
    cross = 0
    crosstimes = []
    xhist = []

    ## euler
    while cross == 0:
        xnew = xold + v * dt
        v += dt*(-gamma*v - dVdx(xnew, k)) + sigma*np.random.normal()
        t += dt
        if np.sign((xnew - threshold)*(xold - threshold)) < 0:
            cross += 1
            crosstimes += [t]
        xold = xnew
        xhist += [xnew]
    time_to_cross = np.diff(np.hstack([0, crosstimes]))
    return cross, time_to_cross, xhist

def feuler_multiple(x0, v0, nsims,
                    k=None, dt=None, sigma=None, gamma=None, threshold=None, seed=None):
    crosses = []
    times = []
    xhists = []
    for i in range(nsims):
        cross, time_to_cross, xhist = feuler_single(x0, v0,
                                                    k=k, dt=dt, sigma=sigma, gamma=gamma,
                                                    threshold=threshold, seed=seed*i)
        crosses += [cross]
        times += [time_to_cross]
        xhists += [xhist]
    crosses = np.hstack(crosses)
    times = np.hstack(times)
    xhists = np.hstack(xhists)
    return np.sum(crosses), times, xhists

def baoab(x0, v0, nsteps,
          k=None, dt=None, tau=None, gamma=None, threshold=None, seed=None):
    """ BAOAB from LM p. 271 with m = 1 """
    ## setup
    np.random.seed(seed)
    q = np.copy(x0)
    q_12 = np.copy(x0)
    q_new = np.copy(x0)
    p = np.copy(v0)
    p_12 = np.copy(v0)
    p_12_hat = np.copy(v0)
    p_new = np.copy(v0)
    h = np.copy(dt)
    zeta = np.random.normal(size=nsteps)
    sigma = np.sqrt(tau * (1-np.exp(-2*gamma*h)))
    t = 0.0
    cross = 0
    crosstimes = []
    xhist = []
    ## baoab
    for i in range(nsteps):
        p_12 = p - h/2 * dVdx(q, k)
        q_12 = q + h/2 * p_12
        p_12_hat = np.exp(-gamma*h) * p_12 + sigma*zeta[i]
        q_new = q_12 + h/2 * p_12_hat
        p_new = p_12_hat - h/2 * dVdx(q_new, k)
        t += dt
        if np.sign((q_new - threshold)*(q - threshold)) < 0:
            cross += 1
            crosstimes += [t]
        q = q_new
        p = p_new
        xhist += [q]
    time_to_cross = np.diff(np.hstack([0, crosstimes]))
    return cross, time_to_cross, xhist

def plot_diagnostics(xhist, k=None, tau=None, dt=None):
    ## histogram / density
    plt.figure(0)
    n, bins, patches = plt.hist(xhist, 100, normed=1)
    xmin = -2.0
    xmax = 2.0
    x = np.arange(xmin, xmax, bins[1]-bins[0])
    P = np.exp(-V(x, k) / tau)
    norm = np.sum(P)*(bins[1]-bins[0])
    plt.plot(x, P/norm)
    plt.xlim([xmin, xmax])

    ## time series
    plt.figure(1)
    plt.plot(np.arange(0, len(xhist)*dt, dt), xhist)
    plt.show()

def compare_results(cross, time_to_cross,
                    x0=None, xt=None, tau=None,
                    nsteps=None, k=None, gamma=None):
    Etau_ana = np.exp(np.abs(V(xt, k) - V(x0, k)) / tau)
    lam_ana = np.exp(-np.abs(V(xt, k) - V(x0, k)) / tau) / (2.0 * np.pi * gamma)
    print("DNS crosses                  : {:d}".format(cross))
    print("DNS crosses / nsteps         : {:0.3e}\n".format(float(cross) / float(nsteps*dt)))

    print("DNS-time mean(time to cross) : {:0.3e}".format(np.mean(time_to_cross)))
    print("ANA-time E[tau]              : {:0.3e}\n".format(Etau_ana))

    print("DNS-rate mean(rate)          : {:0.3e}".format(1.0/np.mean(time_to_cross)))
    print("ANA-rate lambda              : {:0.3e}\n".format(lam_ana))

    a = float(cross) / float(nsteps*dt)
    b = np.mean(time_to_cross)
    c = Etau_ana
    d = 1.0/np.mean(time_to_cross)
    e = lam_ana
    return a, b, c, d, e



## main
## -----------------------------------------------------------------------------
if __name__ == "__main__":
    ## params
    seed = 1234
    tau = 0.15
    tau = 0.35
    dt = 0.01
    k = 1.0
    gamma = 2.0
    sigma = np.sqrt(2.0*dt*gamma*tau)
    threshold = 0.0
    nsteps = int(1e6)
    nsims = 50
    x0 = -1.0
    v0 = 0.1
    xt = threshold
    T = int(dt * nsteps)

    ## one long chain
    ## euler
t0 = time.time()
cross, time_to_cross, xhist = feuler(x0, v0, nsteps,
                                  k=k, dt=dt, sigma=sigma, gamma=gamma,
                                  threshold=threshold, seed=seed)
t1 = time.time()
print(t1-t0)
plot_diagnostics(xhist, k=k, tau=tau, dt=dt)
a, b, c, d, e = compare_results(cross, time_to_cross, x0=x0, xt=xt, tau=tau, nsteps=nsteps, gamma=gamma, k=k)

    ## baoab
t0 = time.time()
cross, time_to_cross, xhist = baoab(x0, v0, nsteps,
                                  k=k, dt=dt, tau=tau, gamma=gamma,
                                  threshold=threshold, seed=seed)
t1 = time.time()
print(t1-t0)
plot_diagnostics(xhist, k=k, tau=tau, dt=dt)
a, b, c, d, e = compare_results(cross, time_to_cross, x0=x0, xt=xt, tau=tau, nsteps=nsteps, gamma=gamma, k=k)


    ## many shorter chains
    cross, time_to_cross, xhist = feuler_multiple(x0, v0, nsims,
                                  k=k, dt=dt, sigma=sigma, gamma=gamma,
                                  threshold=threshold, seed=seed)
    plot_diagnostics(xhist, k=k, tau=tau, dt=dt)
    a, b, c, d, e = compare_results(cross, time_to_cross,
                                    x0=x0, xt=xt, tau=tau, nsteps=len(xhist), gamma=gamma, k=k)

taus = [0.5, 0.4, 0.3, 0.25, 0.225, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.145, 0.14, 0.135, 0.13]
aas = []
bs = []
cs = []
ds = []
es = []
for t in taus:
    print("temperature = {:0.3e}".format(t))
    print("------------------------------------")
    sigma = np.sqrt(2.0*dt*gamma*t)
    # cross, time_to_cross, xhist = feuler(x0, v0, nsteps,
    #                                   k=k, dt=dt, sigma=sigma, gamma=gamma,
    #                                   threshold=threshold, seed=seed)
    cross, time_to_cross, xhist = baoab(x0, v0, nsteps,
                                      k=k, dt=dt, tau=tau, gamma=gamma,
                                      threshold=threshold, seed=seed)
    # cross, time_to_cross, xhist = feuler_multiple(x0, v0, nsims,
    #                               k=k, dt=dt, sigma=sigma, gamma=gamma,
    #                               threshold=threshold, seed=seed)
    a, b, c, d, e = compare_results(cross, time_to_cross,
                                    x0=x0, xt=xt, tau=t, nsteps=len(xhist), gamma=gamma, k=k)
    aas += [a]
    bs += [b]
    cs += [c]
    ds += [d]
    es += [e]

plt.loglog(taus, ds, label="dns")
plt.loglog(taus, es, label="ana")
plt.plot(taus, [ds[i]/es[i] for i in range(len(taus))], label="d/e")
plt.legend()
