using Random
using LinearAlgebra

## -----------------------------------------------------------------------------
## structures
## -----------------------------------------------------------------------------
Copy(x::T) where T = T([deepcopy(getfield(x, k)) for k ∈ fieldnames(T)]...)
mutable struct Cache
    q_12::Float64
    p_12::Float64
    p_12_hat::Float64
    q_old::Float64
    p_old::Float64
    V::Float64
    dVdx::Float64
    zeta::Float64
end
function Cache(; q_12::Float64=0.0,
                 p_12::Float64=0.0, p_12_hat::Float64=0.0,
                 q_old::Float64=0.0, p_old::Float64=0.0,
                 V::Float64=0.0, dVdx::Float64=0.0, zeta::Float64=0.0)
    return Cache(q_12, p_12, p_12_hat, q_old, p_old, V, dVdx, zeta)
end

mutable struct State
    q::Float64  ## position
    p::Float64  ## momentum
    t::Float64  ## time
end
function State(; q::Float64, p::Float64, t::Float64)
    return State(q, p, t)
end

mutable struct Params
    h::Float64
    tau::Float64
    gamma::Float64
    seed::Integer
    k::Float64
    m::Float64
    limit::Float64
end
function Params(; h::Float64, tau::Float64, gamma::Float64, seed::Integer,
                  k::Float64, m::Float64, limit::Float64)
    return Params(h, tau, gamma, seed, k, m, limit)
end

mutable struct Hist
    crosses::Int64
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
function V(x::Float64, k::Float64)
    return k*(1-x^2)^2
end
function dVdx(x::Float64, k::Float64)
    return -4*k*x*(1-x^2)
end
function V!(out::Float64, x::Float64, k::Float64)
    out = k*(1-x^2)^2
end
function dVdx!(out::Float64, x::Float64, k::Float64)
    our = -4*k*x*(1-x^2)
end
## -----------------------------------------------------------------------------


## -----------------------------------------------------------------------------
## integrate
## -----------------------------------------------------------------------------
function euler_fix(S::State, N::Int64; sigma::Float64, P::Params, C::Cache,
                                       max_cross::Int64=Int64(1e4))
    ## extract
    q = S.q; p = S.p; t = S.t; q_old = S.q
    h = P.h; gamma = P.gamma; k = P.k
    f = C.dVdx

    ## setup
    j::Int64 = 1
    crosses::Int64 = 0
    qhist = zeros(Float64, N)
    times = zeros(Float64, 1000)
    zetas = randn(N)

    ## integrate
    for i = 1:N
        ## euler update
        q += p .* h
        p += h .* (-gamma .* p .- dVdx!(f, q, k)) .+ (sigma .* zetas[i])

        ## check
        if (q - limit)*(q_old - limit) < 0.0
            j += 1
            crosses +=1
            times[j] = t
        end

        ## update
        q_old = q
        t += h
        qhist[i] = q
    end

    ## return
    H = Hist(crosses, qhist, diff([0; times[1:j]]))
    return H
end
function baoab_fix(S::State, N::Int64; sigma::Float64, P::Params, C::Cache,
                                       max_cross::Int64=Int64(1e4))
    ## extract
    q = S.q; p = S.p; t = S.t; q_old = S.q
    h = P.h; gamma = P.gamma; k = P.k
    f = C.dVdx

    ## setup
    j::Int64 = 1
    crosses::Int64 = 0
    qhist = zeros(Float64, N)
    times = zeros(Float64, 1000)
    zetas = randn(N)

    ## integrate
    for i = 1:N
        ## baoab update
        p_12 = p - h/2 * dVdx!(f, q, k)
        q_12 = q + h/2 * p_12 / m
        p_12_hat = exp(-gamma * h) * p_12 + sigma*zetas[i]
        q = q_12 + h/2 * p_12_hat / m
        p = p_12_hat - h/2 .* dVdx!(f, q, k)

        ## check
        if (q - limit)*(q_old - limit) < 0.0
            j += 1
            crosses +=1
            times[j] = t
        end

        ## update
        q_old = q
        t += h
        qhist[i] = q
    end

    ## return
    H = Hist(crosses, qhist, diff([0; times[1:j]]))
    return H
end
function integrator_fix(SS::State, N::Int64; PP::Params, CC::Cache, ut::Symbol)
    """
    N: number of steps
    limit: q-threshold
    tb: traceback option (T/F)
    ut: update type option (:euler!, :baoab!)
    """
    ## setup
    S = Copy(SS)
    P = Copy(PP)
    Random.seed!(P.seed)

    ## sigma & integrate
    if ut == :euler!
        sigma = sqrt(2.0 * P.h * P.gamma * P.tau)
        H = euler_fix(S::State, N::Int64; sigma=sigma, P=P, C=CC)
    elseif ut == :baoab!
        sigma = sqrt(P.tau * (1.0 - exp(-2.0 * P.gamma * P.h)))
        H = baoab_fix(S::State, N::Int64; sigma=sigma, P=P, C=CC)
    end
    return H
end
