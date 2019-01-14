## -----------------------------------------------------------------------------
## simulations
## -----------------------------------------------------------------------------
seed = 1234
tau = 0.25
h = 0.01
k = 1.0
m = 1.0
gamma = 2.0
limit = 0.0
nsteps = Int64(1e7)
nsims = 50
q0 = -1.0
p0 = 0.1
qt = copy(limit)
S = State(q=q0, p=p0, t=0.0)
C = Cache()
P = Params(seed=seed, tau=tau, h=h, k=k, m=m, gamma=gamma, limit=limit)

@time H = integrator_fix(S, nsteps; PP=P, CC=C, ut=:baoab!)
@time H = integrator_fix(S, nsteps; PP=P, CC=C, ut=:euler!)
