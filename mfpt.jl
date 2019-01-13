## -----------------------------------------------------------------------------
## structures
## -----------------------------------------------------------------------------
Copy(x::T) where T = T([deepcopy(getfield(x, k)) for k ∈ fieldnames(T)]...)
mutable struct Cache
    q_12::Float64
    q_new::Float64
    p_12::Float64
    p_12_hat::Float64
    p_new::Float64
    V::Float64
    dVdx::Float64
    zeta::Float64
end
function Cache(; q_12::Float64=0.0, q_new::Float64=0.0,
                 p_12::Float64=0.0, p_12_hat::Float64=0.0, p_new::Float64=0.0,
                 V::Float64=0.0, dVdx::Float64=0.0, zeta::Float64=0.0)
    return Cache(q_12, q_new, p_12, p_12_hat, p_new, V, dVdx, zeta)
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
end
function Params(; h::Float64, tau::Float64, gamma::Float64, seed::Integer, k::Float64, m::Float64)
    return Params(h, tau, gamma, seed, k, m)
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
# function baoab!(S::State, sigma::Float64, zeta::Float64; P::Params, C::Cache)
#     """
#     see L-M book p. 271
#     """
#     C.p_12 = S.p - P.h/2 * dVdx(S.q, P. k)
#     C.q_12 = S.q + P.h/2 * C.p_12 / P.m
#     C.p_12_hat = exp(-P.gamma * P.h) * C.p_12 + sigma*zeta
#     C.q_new = C.q_12 + P.h/2 * C.p_12_hat / P.m
#     C.p_new = C.p_12_hat - P.h/2 * dVdx(C.q_new, P.k)
#
#     S.q = C.q_new
#     S.p = C.p_new
#     S.t += P.h
# end
# function euler!(S::State, sigma::Float64, zeta::Float64; P::Params, C::Cache)
#     S.q += S.p * P.h
#     S.p += P.h*(-P.gamma * S.p - dVdx!(C.dVdx, S.q, P.k)) + sigma*zeta
#     S.t += P.h
# end
function euler!(q::Float64, p::Float64, t::Float64, sigma::Float64, zeta::Float64; P::Params, C::Cache)
    q += p .* P.h
    p += P.h*(-P.gamma .* p .- dVdx(q, P.k)) .+ sigma.*zeta
    t += P.h
end
function euler!(q::Float64, p::Float64, t::Float64, sigma::Float64, zeta::Float64; P::Params, C::Cache)
    q += p .* P.h
    p += P.h.*(-P.gamma .* p .- dVdx(q, P.k)) .+ sigma.*zeta
    t += P.h
end
function baoab!(q::Float64, p::Float64, t::Float64, sigma::Float64, zeta::Float64; P::Params, C::Cache)
    """
    see L-M book p. 271
    """
    C.p_12 = p - P.h/2 * dVdx(q, P. k)
    C.q_12 = q + P.h/2 * C.p_12 / P.m
    C.p_12_hat = exp(-P.gamma * P.h) * C.p_12 + sigma*zeta
    C.q_new = C.q_12 + P.h/2 * C.p_12_hat / P.m
    C.p_new = C.p_12_hat - P.h/2 * dVdx(C.q_new, P.k)

    q = C.q_new
    p = C.p_new
    t += P.h
end
function integrator_fix(SS::State, N::Integer, limit::Float64;
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
    # H = Hist(N)
    crosses = 0
    qhist = zeros(N)
    times = zeros(N)
    zetas = randn(N)
    q = copy(S.q)
    p = copy(S.p)
    t = copy(S.t)
    q_old = copy(S.q)

    ## sigma
    if ut == :euler!
        sigma = sqrt(2.0 * P.h * P.gamma * P.tau)
    elseif ut == :baoab!
        sigma = sqrt(P.tau * (1.0 - exp(-2.0 * P.gamma * P.h)))
    end
    ## output
    j = 1
    # H.qhist[1] = S.q
    qhist[1] = S.q

    ## integrate euler
    if ut == :euler!
        for i = 2:N
            # q_old = S.q
            q_old = q
            # euler!(S, sigma, zetas[i]; P=P, C=C)
            # euler!(q,p,t, sigma, zetas[i]; P=P, C=C)

            # q += p .* P.h
            # p += P.h .* (-P.gamma .* p .- dVdx(q, P.k)) .+ sigma.*zetas[i]
            q += p * P.h
            p += P.h * (-P.gamma * p - dVdx(q, P.k)) + sigma*zetas[i]
            t += P.h

            # H.qhist[i] = S.q
            # qhist[i] = S.q
            qhist[i] = q
            ## check & update crossing event
            # if sign((S.q - limit)*(q_old - limit)) < 0
            if sign((q - limit)*(q_old - limit)) < 0
                j += 1
                # H.crosses += 1
                # H.times[j] = S.t
                crosses +=1
                # times[j] = S.t
                times[j] = t
            end
        end
    elseif ut == :baoab!
        for i = 2:N
            baoab!(S, sigma, zetas[i]; P=P, C=C)
            H.qhist[i] = S.q
            ## check & update crossing event
            if sign((H.qhist[i] - limit)*(H.qhist[i-1] - limit)) < 0
                j += 1
                H.crosses += 1
                H.times[j] = S.t
            end
        end
    end
    # H.times = diff([0; H.times[1:j]])
    H = Hist(crosses, qhist, diff([0; times[1:j]]))
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
nsteps = Int(1e7)
nsims = 50
q0 = -1.0
p0 = 0.1
qt = copy(limit)
S = State(q=q0, p=p0, t=0.0)
C = Cache()
P = Params(seed=seed, tau=tau, h=h, k=k, m=m, gamma=gamma)

@time H = integrator_fix(S, nsteps, limit; PP=P, CC=C, ut=:euler!)
