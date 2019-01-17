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
    # return k .* (1.0 .- x.^2).^2
    return 0.5*k .* (x .+ 1.0).^2
end
function dVdx(x, k)
    # return (-4.0 * k) .* x .* (1.0 .- x.^2)
    return k .* (x .+ 1.0)
end
function d2Vdx2(out::Float64, x::Float64, k::Float64)
    # return (12.0 * k) .* (x.^2 .- 1.0/3.0)
    return k
end
function V!(out::Float64, x::Float64, k::Float64)
    # out = k .* (1.0 .- x.^2).^2
    out = 0.5*k .* (x .+ 1.0).^2
end
function dVdx!(out::Float64, x::Float64, k::Float64)
    # out = (-4.0 * k) .* x .* (1.0 .- x.^2)
    out = k .* (x .+ 1.0)
end
function d2Vdx2!(out::Float64, x::Float64, k::Float64)
    # out = (12.0 * k) .* (x.^2 .- 1.0/3.0)
    out = k
end
function hh(x, limit)
    return x - limit
end
function dhhdx(x, limit)
end

function analytic(Sbar::State, Sstar::State, P::Params)
    ## setup
    qbar = Sbar.q
    qstar = Sstar.q
    Gstar = dVdx(qstar, P.k)
    gstar = dCdx(qstar, P.k)

    ## hessians
    Hbar = energy_h(xbar, theta)                   ## energy hess at x_eq0
    Hstar = energy_h(xstar, theta)                 ## energy hess at x_eqc
    hstar = constraint_h(xstar, theta, l, c_type)  ## constraint hess at x_eqc

    ## calc B
    n = gstar / norm(gstar)
    n = reshape(n, length(xbar), 1)
    E = nullspace(Matrix(n'))
    B = det(E'*(Hstar - k*hstar)*E) * (norm(Gstar)^2 / det(E'*E))
    @assert(B > 0)

    pf = P.gamma *1
end
## -----------------------------------------------------------------------------


## -----------------------------------------------------------------------------
## integrate for fixed time horizon
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
    qhist[1] = q

    ## integrate
    for i = 2:N
        ## euler update
        q += p .* h
        p += h .* (-gamma .* p .- dVdx!(f, q, k)) .+ (sigma .* zetas[i])

        ## check
        if (q - limit)*(q_old - limit) < 0.0
            j += 1
            crosses += 1
            times[j] = t + h
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
function euler2_fix(S::State, N::Int64; sigma::Float64, P::Params, C::Cache,
                                       max_cross::Int64=Int64(1e4))
    """
    example which has wrong asymptotics; note: the h factor on the noise is weird
    """
    ## extract
    q = S.q; p = S.p; t = S.t; q_old = copy(S.q)
    h = P.h; gamma = P.gamma; k = P.k
    f = C.dVdx

    ## setup
    j::Int64 = 1
    crosses::Int64 = 0
    qhist = zeros(Float64, N)
    times = zeros(Float64, max_cross)
    zetas = randn(N)
    qhist[1] = q

    ## integrate
    for i = 2:N
        ## euler update
        # q += (-dVdx!(f, q, k) * h^2 + gamma * q * h) / (1.0 + gamma*h) + (sigma .* zetas[i])
        # q = (h^2*(-dVdx!(f, q, k) + gamma*q/h) + 2.0*q - q_old) / (1.0 - gamma*h) + h^(3/5)*(sigma .* zetas[i])
        q = (h^2*(-dVdx!(f, q, k) + gamma*q/h) + 2.0*q - q_old) * (1.0 - gamma*h) + sqrt(h)*(sigma .* zetas[i])

        ## check
        if (q - limit)*(q_old - limit) < 0.0
            j += 1
            crosses += 1
            times[j] = t
        end

        ## update
        q_old = q
        t += h + h
        qhist[i] = q
    end

    ## return
    H = Hist(crosses=crosses, qhist=qhist, times=diff([0; times[1:j]]))
    return H
end
function euler3_fix(S::State, N::Int64; sigma::Float64, P::Params, C::Cache,
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
    qhist[1] = q

    ## integrate
    for i = 2:N
        ## euler update
        q += h .* p .+ sqrt(h) * (sigma .* zetas[i])
        p += h .* (-gamma .* p .- dVdx!(f, q, k))

        ## check
        if (q - limit)*(q_old - limit) < 0.0
            j += 1
            crosses += 1
            times[j] = t + h
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
    qhist[1] = q

    ## integrate
    for i = 2:N
        ## baoab update
        p_12 = p - h/2 * dVdx!(f, q, k)
        q_12 = q + h/2 * p_12 / m
        p_12_hat = exp(-gamma * h) * p_12 + sigma*zetas[i]
        q = q_12 + h/2 * p_12_hat / m
        p = p_12_hat - h/2 .* dVdx!(f, q, k)

        ## check
        if (q - limit)*(q_old - limit) < 0.0
            j += 1
            crosses += 1
            times[j] = t + h
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
    elseif ut == :euler2!
        sigma = sqrt(2.0 * P.h * P.gamma * P.tau)
        H = euler2_fix(S::State, N::Int64; sigma=sigma, P=P, C=CC, max_cross=max_cross)
    elseif ut == :euler3!
        sigma = sqrt(2.0 * P.h * P.gamma * P.tau)
        H = euler3_fix(S::State, N::Int64; sigma=sigma, P=P, C=CC, max_cross=max_cross)
    end
    return H
end
## -----------------------------------------------------------------------------


## -----------------------------------------------------------------------------
## integrate for variable time horizon
## -----------------------------------------------------------------------------
function euler_fix_fail(S::State, N::Int64; sigma::Float64, P::Params, C::Cache)
    """
    run `N` steps of euler until failure
    """
    ## extract
    q = S.q; p = S.p; t = S.t; q_old = S.q
    h = P.h; gamma = P.gamma; k = P.k; limit = P.limit
    f = C.dVdx

    ## setup
    j::Int64 = 1
    fail::Int64 = 0
    qhist = zeros(Float64, N)
    time::Float64 = 0.0
    zetas = randn(N)
    qhist[1] = q

    ## integrate
    for i = 2:N
        ## euler update
        q += p .* h
        p += h .* (-gamma .* p .- dVdx!(f, q, k)) .+ (sigma .* zetas[i])

        ## check
        if hh(q, limit) > 0.0
            fail = 1
            time = t + h
            qhist[i] = q
            Sout = State(q=q, p=p, t=time)
            j = i
            break
        end

        ## update
        q_old = q
        t += h
        qhist[i] = q
    end

    ## return
    Sout = State(q=q, p=p, t=time)
    return fail, Sout, qhist[1:j]
end
function euler_var(S::State, N::Int64; sigma::Float64, P::Params, C::Cache, Nchunk::Int64=Int64(1e7))
    ## setup
    SS = Copy(S)
    crosses::Int64 = 0
    qhist = zeros(Float64, 1)
    times = Float64[]
    fail::Int64 = 0
    time::Float64 = 0.0

    ## integrate
    for i = 1:N
        while fail == 0
            fail, SS, hist = euler_fix_fail(SS::State, Nchunk::Int64; sigma=sigma, P=P, C=C)
            push!(qhist, hist...)
            time = SS.t
        end
        SS = Copy(S)
        push!(times, time)
        crosses += fail
        fail = 0
    end

    ## return
    H = Hist(crosses=crosses, qhist=qhist, times=times)
    return H
end
function integrator_var(SS::State, Nfail::Int64; PP::Params, CC::Cache, ut::Symbol)
    """
    Nfail: number of failures
    limit: q-threshold
    ut: update type option (:euler!, :baoab!)
    """
    ## setup
    S = Copy(SS)
    P = Copy(PP)
    Random.seed!(P.seed)

    ## sigma & integrate
    if ut == :euler!
        sigma = sqrt(2.0 * P.h * P.gamma * P.tau)
        H = euler_var(S::State, Nfail::Int64; sigma=sigma, P=P, C=CC)
    end
    return H
end

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
    println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    out = Dict()
    out[:Etau_dns_1] = Etau_dns_1
    out[:Etau_dns_2] = Etau_dns_2
    out[:Etau_ana] = Etau_ana
    out[:lam_dns] = lam_dns
    out[:lam_ana] = lam_ana
    return out
end
## -----------------------------------------------------------------------------
