using Random
using LinearAlgebra
using Statistics
using PyPlot
using Formatting


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
function Hist(; crosses::Int64, times::Array{Float64,1}, qhist::Array{Float64,1})
    return Hist(crosses, times, qhist)
end
## -----------------------------------------------------------------------------


## -----------------------------------------------------------------------------
## functions
## -----------------------------------------------------------------------------
function V(x, k)
    return k .* (1.0 .- x.^2).^2
end
function dVdx(x, k)
    return -4.0 .* k .* x .* (1 .- x.^2)
end
function V!(out::Float64, x::Float64, k::Float64)
    out = k .* (1.0 .- x.^2).^2
end
function dVdx!(out::Float64, x::Float64, k::Float64)
    our = -4.0 .* k .* x .* (1 .- x.^2)
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
    times = zeros(Float64, max_cross)
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
    H = Hist(crosses=crosses, qhist=qhist, times=diff([0; times[1:j]]))
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
    times = zeros(Float64, max_cross)
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
    H = Hist(crosses=crosses, qhist=qhist, times=diff([0; times[1:j]]))
    return H
end
function integrator_fix(SS::State, N::Int64; PP::Params, CC::Cache, ut::Symbol,
                                             max_cross::Int64=Int64(1e4))
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
        H = euler_fix(S::State, N::Int64; sigma=sigma, P=P, C=CC, max_cross=max_cross)
    elseif ut == :baoab!
        sigma = sqrt(P.tau * (1.0 - exp(-2.0 * P.gamma * P.h)))
        H = baoab_fix(S::State, N::Int64; sigma=sigma, P=P, C=CC, max_cross=max_cross)
    end
    return H
end
## -----------------------------------------------------------------------------


## -----------------------------------------------------------------------------
## diagnostics
## -----------------------------------------------------------------------------
function plot_diagnostics(H::Hist, P::Params; subsample=1000)
    ## histogram / density
    fig = figure(figsize=(10, 8))
    ax1 = fig[:add_subplot](2,1,1)
    n, bins, patches = ax1[:hist](H.qhist, 100, normed=1, label="numerical");
    println(patches)
    dd = bins[2] - bins[1]
    xmin = -2.0
    xmax = 2.0
    q = collect(range(xmin, stop=xmax, step=dd))
    PP = exp.(-V(q, P.k) / P.tau)
    den = sum(PP) * dd
    ax1[:plot](q, PP ./ den, label="analytic")
    ax1[:set_xlim](xmin, xmax)
    legend()

    ## time series
    ax2 = fig[:add_subplot](2,1,2)
    ts = collect(range(P.h, stop=length(H.qhist)*P.h, step=P.h))[1:subsample:end]
    qs = H.qhist[1:subsample:end]
    ax2[:plot](ts, qs, label="position time-series")
    legend()
    fig[:show]()
end
function compare_results(H::Hist, P::Params; S0::State)
    q0 = S0.q
    qt = P.limit
    Etau_ana = exp(abs(V(qt, P.k) - V(q0, P.k)) / P.tau)
    lam_ana = exp(-abs(V(qt, P.k) - V(q0, P.k)) / P.tau) / (2.0 * pi * P.gamma)

    Etau_dns_1 = (H.crosses / length(H.qhist)) / P.h
    Etau_dns_2 = mean(H.times)
    lam_dns = 1.0 / Etau_dns_2

    println("crosses ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    printfmt("DNS crosses                                          : {:d}\n", H.crosses)
    printfmt("DNS mean(time to cross) = (crosses / nsteps) / dt    : {:0.3e}\n\n", Etau_dns_1)

    println("exit time ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    printfmt("DNS-time mean(time to cross)                         : {:0.3e}\n", Etau_dns_2)
    printfmt("ANA-time E[tau] = e^(|V(qt) - V(q0)| / tau)          : {:0.3e}\n", Etau_ana)
    printfmt("exit time ratio DNS/ANA                              : {:0.3e}\n\n", Etau_dns_2/Etau_ana)

    println("exit rate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    printfmt("DNS-rate mean(rate)                                  : {:0.3e}\n", lam_dns)
    printfmt("ANA-rate lam = 1/(2pi*gam) e^(|V(qt) - V(q0)| / tau) : {:0.3e}\n", lam_ana)
    printfmt("exit rate ratio DNS/ANA                              : {:0.3e}\n", lam_dns/lam_ana)
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    out = Dict()
    out[:Etau_dns_1] = Etau_dns_1
    out[:Etau_dns_2] = Etau_dns_2
    out[:Etau_ana] = Etau_ana
    out[:lam_dns] = lam_dns
    out[:lam_ana] = lam_ana
    return out
end
