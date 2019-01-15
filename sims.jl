## -----------------------------------------------------------------------------
## simulations
## -----------------------------------------------------------------------------
include("mfpt.jl")
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

taus = [0.5, 0.4, 0.3, 0.25, 0.225, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.145, 0.14, 0.135, 0.13]
Hs = Array{Hist}(undef, length(taus))
outs = Array{Dict}(undef, length(taus))
for z in zip(taus, 1:length(taus))
    t = z[1]
    i = z[2]
    printfmt("tau: {:0.3e} ======================================================", t)
    P.tau = t
    @time H = integrator_fix(S, nsteps; PP=P, CC=C, ut=:euler!, max_cross=Int64(1e5))
    out = compare_results(H, P; S0=S0)
    Hs[i] = H
    outs[i] = out
end
