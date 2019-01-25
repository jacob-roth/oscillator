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
    phist::Array{Float64,1}
end
function Hist(; crosses::Int64, times::Array{Float64,1}, qhist::Array{Float64,1}, phist::Array{Float64,1})
    return Hist(crosses, times, qhist, phist)
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
function d2Vdx2(x::Float64, k::Float64)
    return k
end
function d2Vdx2!(out::Float64, x::Float64, k::Float64)
    # out = (12.0 * k) .* (x.^2 .- 1.0/3.0)
    out = k
end
function hh(q, p, limit)
    return q - limit
    # return p - limit
end
function dhhdx(q, p, limit)
    return [1.0; 0.0]
end

function analytic(Sbar::State, Sstar::State, P::Params)
    ## setup
    qbar = Sbar.q
    qstar = Sstar.q
    pbar = Sbar.p
    pstar = Sstar.p
    Gstar = [dVdx(qstar, P.k); 0.0]
    gstar = dhhdx(qstar, pstar, P.limit)
    S = [1.0 0.0; 0.0 0.0]

    ## hessians
    Hbar = d2Vdx2(qbar, P.k)                  ## energy hess at x_eq0
    Hstar = d2Vdx2(qstar, P.k)                ## energy hess at x_eqc
    hstar = 0.0                               ## constraint hess at x_eqc

    ## calc B
    n = gstar ./ norm(gstar)
    n = reshape(n, length(n), 1)
    E = nullspace(Matrix(n'))
    B = det(E'*(Hstar - k*hstar)*E) * (norm(Gstar)^2 / det(E'*E))
    @assert(B > 0)

    ## calc rate
    pf = P.gamma .* Gstar'*S*Gstar
    pf *= sqrt(det(Hbar) / (2*pi*P.tau*abs(B)))
    estar = V(qstar, P.k)
    ebar = V(qbar, P.k)
    ef = exp(-( estar - ebar ) / P.tau)
    lam = pf * ef
    return [pf, ef, lam]
end
## -----------------------------------------------------------------------------


## -----------------------------------------------------------------------------
## integrate for fixed time horizon
## -----------------------------------------------------------------------------
function euler_fix(S::State, N::Int64; sigma::Float64, P::Params, C::Cache,
                                       max_cross::Int64=Int64(1e4))
    ## extract
    q = S.q; p = S.p; t = S.t; q_old = S.q; p_old = S.p
    h = P.h; gamma = P.gamma; k = P.k
    f = C.dVdx

    ## setup
    j::Int64 = 1
    crosses::Int64 = 0
    qhist = zeros(Float64, N)
    phist = zeros(Float64, N)
    times = zeros(Float64, max_cross)
    zetas = randn(N)
    etas = randn(N)
    qhist[1] = q
    phist[1] = p

    ## integrate
    for i = 2:N
        ## euler update
        # q += p .* h
        # p += h .* (-gamma .* p .- dVdx!(f, q, k)) .+ (sigma .* zetas[i])
        q += h .* (-gamma*dVdx!(f, q, k) .+ p) .+ (sigma .* etas[i])
        p += h .* (- dVdx!(f, q, k))

        ## check
        # if (q - limit)*(q_old - limit) < 0.0
        if (hh(q, p, limit) >= 0.0) && (hh(q_old, p_old, limit) < 0.0)
            j += 1
            crosses += 1
            times[j] = t + h
        end

        ## update
        q_old = q
        p_old = p
        t += h
        qhist[i] = q
        phist[i] = p
    end

    ## return
    H = Hist(crosses=crosses, qhist=qhist, times=diff(times[1:j]), phist=phist)
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
    phist = zeros(Float64, N)
    time::Float64 = 0.0
    zetas = randn(N)
    etas = randn(N)
    qhist[1] = q

    ## integrate
    for i = 2:N
        ## euler update
        # q += p .* h
        # p += h .* (-gamma .* p .- dVdx!(f, q, k)) .+ (sigma .* zetas[i])
        q += h .* (-gamma*dVdx!(f, q, k) .+ p) .+ (sigma .* etas[i])
        p += h .* (- dVdx!(f, q, k))

        ## check
        if hh(q, p, limit) >= 0.0
            fail = 1
            time = t + h
            qhist[i] = q
            phist[i] = p
            Sout = State(q=q, p=p, t=time)
            j = i
            break
        end

        ## update
        q_old = q
        t += h
        qhist[i] = q
        phist[i] = p
    end

    ## return
    Sout = State(q=q, p=p, t=time)
    return fail, Sout, qhist[1:j], phist[1:j]
end
function euler_var(S::State, N::Int64; sigma::Float64, P::Params, C::Cache, Nchunk::Int64=Int64(1e7))
    ## setup
    SS = Copy(S)
    crosses::Int64 = 0
    qhist = Float64[]
    phist = Float64[]
    times = Float64[]
    fail::Int64 = 0
    time::Float64 = 0.0

    ## integrate
    for i = 1:N
        while fail == 0
            fail, SS, histq, histp = euler_fix_fail(SS::State, Nchunk::Int64; sigma=sigma, P=P, C=C)
            push!(qhist, histq...)
            push!(phist, histp...)
            time = SS.t
        end
        SS = Copy(S)
        # SS.q = sigma*randn() - 1.0
        # SS.p = sigma*randn()
        push!(times, time)
        crosses += fail
        fail = 0
    end

    ## return
    H = Hist(crosses=crosses, qhist=qhist, phist=phist, times=times)
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
    ax1 = fig[:add_subplot](3,1,1)
    n, bins, patches = ax1[:hist](H.qhist, 100, normed=1, label="numerical");
    println(patches)
    dd = bins[2] - bins[1]
    xmin = -2.0
    xmax = 2.0
    q = collect(range(xmin, stop=xmax, step=dd))

    PP = P.gamma / (2.0 * pi * P.tau) .* exp.(-V(q, P.k) / P.tau)
    Sbar = State(q=-1.0, p=0.0, t=0.0)
    Sstar = State(q=0.0, p=0.0, t=0.0)
    lam_ana = analytic(Sbar, Sstar, P)[3][1]
    # if P.gamma <= 0.01
    #     ## NOTE: not sure how to handle in underdamped case
    #     PP = P.gamma / (2.0 * pi * P.tau) * exp.(-V(q, P.k) / P.tau)
    #     lam_ana = P.gamma * exp(-abs(V(P.limit, P.k) - V(-1, P.k)) / P.tau) / (2.0 * pi * P.tau)
    # elseif (P.gamma > 0.01) && (P.gamma < 10.0)
    #     ## NOTE: prefactor gets washed out since constant
    #     PP = (2.0*pi)^(-1) * exp.(-V(q, P.k) / P.tau)
    #     lam_ana = exp(-abs(V(P.limit, P.k) - V(-1, P.k)) / P.tau) / (2.0 * pi)
    # elseif P.gamma >= 10.0
    #     PP = exp.(-V(q, P.k) / P.tau)
    #     lam_ana = exp(-abs(V(P.limit, P.k) - V(-1, P.k)) / P.tau) / (2.0 * pi * P.gamma)
    # end

    den = sum(PP) * dd
    ax1[:plot](q, PP ./ den, label="analytic")
    ax1[:set_xlim](xmin, xmax)
    legend()

    ## time series
    ax2 = fig[:add_subplot](3,1,2)
    ts = collect(range(P.h, stop=length(H.qhist)*P.h, step=P.h))[1:subsample:end]
    qs = H.qhist[1:subsample:end]
    ax2[:plot](ts, qs, label="position time-series")
    legend()

    ax3 = fig[:add_subplot](3,1,3)
    ax3[:hist](H.times, density=true)
    yscale("log", nonposy="clip")
    lam_dns = 1.0 / mean(H.times)
    x = collect(range(0, stop=maximum(H.times), length=1000))
    ax3[:plot](x, lam_ana .* exp.(-lam_ana .* x), label="analytic = $(round.(lam_ana, digits=5))")
    ax3[:plot](x, lam_dns .* exp.(-lam_dns .* x), label="dns = $(round.(lam_dns, digits=5))")
    legend()
    fig[:show]()
end
function compare_results(H::Hist, P::Params; S0::State)
    q0 = S0.q
    qt = P.limit
    Etau_ana = exp(abs(V(qt, P.k) - V(q0, P.k)) / P.tau)
    ## TODO: should have three separate analytic calcs depending on gamma?
    # lam_ana = exp(-abs(V(qt, P.k) - V(q0, P.k)) / P.tau) / (2.0 * pi * P.gamma)

    Sbar = State(q=-1.0, p=0.0, t=0.0)
    Sstar = State(q=0.0, p=0.0, t=0.0)
    lam_ana = analytic(Sbar, Sstar, P)[3][1]

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
