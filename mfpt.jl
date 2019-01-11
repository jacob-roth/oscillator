## -----------------------------------------------------------------------------
## structures
## -----------------------------------------------------------------------------
Copy(x::T) where T = T([deepcopy(getfield(x, k)) for k ∈ fieldnames(T)]...)
mutable struct Cache
    q_12::Real
    q_new::Real
    p_12::Real
    p_12_hat::Real
    p_new::Real
    V::Real
    dVdx::Real
    zeta::Real
end
function Cache(; q_12::Real=0.0, q_new::Real=0.0,
                 p_12::Real=0.0, p_12_hat::Real=0.0, p_new::Real=0.0,
                 V::Real=0.0, dVdx::Real=0.0, zeta::Real=0.0)
    return Cache(q_12, q_new, p_12, p_12_hat, p_new, V, dVdx, zeta)
end

mutable struct State
    q::Real  ## position
    p::Real  ## momentum
    t::Real  ## time
end
function State(; q::Real, p::Real, t::Real)
    return State(q, p, t)
end

mutable struct Params
    h::Real
    tau::Real
    gamma::Real
    seed::Integer
    k::Real
    m::Real
end
function Params(; h::Real, tau::Real, gamma::Real, seed::Integer, k::Real, m::Real)
    return Params(h, tau, gamma, seed, k, m)
end

mutable struct Hist
    crosses::Integer
    times::Array{Float64,1}
    qhist::Array{Float64,1}
end
function Hist(nsteps)
    return Hist(0, zeros(nsteps), zeros(nsteps))
end
## -----------------------------------------------------------------------------


## -----------------------------------------------------------------------------
## functions
## -----------------------------------------------------------------------------
function V(x, k)
    return k*(1-x^2)^2
end

function dVdx(x, k)
    return -4*k*x*(1-x^2)
end
## -----------------------------------------------------------------------------


## -----------------------------------------------------------------------------
## integrate
## -----------------------------------------------------------------------------
function baoab!(S::State, sigma::Real, zeta::Real; P::Params, C::Cache)
    """
    see L-M book p. 271
    """
    C.p_12 = S.p - P.h/2 * dVdx(S.q, P. k)
    C.q_12 = S.q + P.h/2 * C.p_12 / P.m
    C.p_12_hat = exp(-P.gamma * P.h) * C.p_12 + sigma*zeta
    C.q_new = C.q_12 + P.h/2 * C.p_12_hat / P.m
    C.p_new = C.p_12_hat - P.h/2 * dVdx(C.q_new, P.k)

    S.q = C.q_new
    S.p = C.p_new
    S.t += P.h
end
function euler!(S::State, sigma::Real, zeta::Real; P::Params, C::Cache)
    S.q += S.p * P.h
    S.p += P.h*(-P.gamma * S.p - dVdx(S.q, P.k)) + sigma*zeta
    S.t += P.h
end
function integrator_fix(SS::State, N::Integer, limit::Real;
                        PP::Params, CC::Cache, ut::Symbol)
    """
    N: number of steps
    limit: q-threshold
    tb: traceback option (T/F)
    ut: update type option (:euler!, :baoab!)
    """
    ## setup
    S = Copy(SS)
    P = Copy(PP)
    H = Hist(N)
    zetas = randn(N)

    ## sigma
    if ut == :euler!
        sigma = sqrt(2.0 * P.h * P.gamma * P.tau)
    elseif ut == :baoab!
        sigma = sqrt(P.tau * (1.0 - exp(-2.0 * P.gamma * P.h)))
    end
    ## output
    j = 1
    H.qhist[1] = S.q

    ## integrate
    for i = 2:N
        ## step
        eval(ut)(S, sigma, zetas[i]; P=P, C=C)
        ## update
        H.qhist[i] = S.q
        ## check & update crossing event
        if sign((H.qhist[i] - limit)*(H.qhist[i-1] - limit)) < 0
            j += 1
            println("cross")
            println(j)
            println(S.t)
            println(H.times[j-1])
            println("")
            H.crosses += 1
            H.times[j] = S.t
        end
    end
    H.times = diff([0; H.times[1:j]])
    return H
end


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
nsteps = Int(1e6)
nsims = 50
q0 = -1.0
p0 = 0.1
qt = copy(limit)
S = State(q=q0, p=p0, t=0.0)
C = Cache()
P = Params(seed=seed, tau=tau, h=h, k=k, m=m, gamma=gamma)

@time H = integrator_fix(S, nsteps, limit; PP=P, CC=C, ut=:euler!)
