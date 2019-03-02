mutable struct Link
    ID::Int64       ## dimension ID
    head::Int64     ## originates from this ID
    tail::Int64     ## ends at this ID
    limit::Float64  ## constraint limit
end

mutable struct State
    ## TODO: how to add magnitude
    q::Array{Float64,1}  ## like angle
    p::Array{Float64,1}  ## like frequency
    L::Array{Link,1}     ## like tline branches
end
function State(q, p, L::Array{Link,1})
    lIDs = [l.ID for l in L]
    @assert(sort(lIDs) == lIDs)
    return State(q, p, L)
end

mutable struct Energy
    pe::Float64
    ke::Float64
    te::Float64
end

mutable struct ConEnergy
    e::Array{Float64,1}  ## each dimension's energy
end

mutable struct Cache
    S::State
    n::Int64                  ## dimension `q` and `p`
    N::Int64                  ## system dimension `q` + `p`
    NL::Int64                 ## system link dimension
    E::Energy                 ## hamiltonian energy
    CE::ConEnergy             ## constraint energy
    A::AbstractArray          ## prefactor dynamics
    Y::AbstractArray          ## "admittance" psd with `1` ∈ kernel
    grad_q::Array{Float64,1}  ## dUdq
    grad_p::Array{Float64,1}  ## dUdp
    grad::Array{Float64,1}    ## gradient = [dUdq; dUdp]
    Agrad::Array{Float64,1}   ## `A` * `grad`
    sigma::Array{Float64,1}   ## noise coeff
    dW::Array{Float64,1}      ## noise
    h::Float64
end
function Cache(S::State)
    n = length(S.q); @assert(n == length(S.p))
    N = 2n
    NL = length(S.L)
    E = Energy(0.0, 0.0, 0.0)
    CE = ConEnergy(zeros(n))
    A = [spzeros(n, n)           spdiagm(0 => ones(n));
         spdiagm(0 => -ones(n))          spzeros(n, n)]
    Y = spdiagm(0 => ones(n))
    grad_q = zeros(n)
    grad_p = zeros(n)
    grad = zeros(N)
    Agrad = zeros(N)
    sigma = zeros(N)
    dW = zeros(N)
    h = 0.0
    return Cache(S, n, N, NL, E, CE, A, Y, grad_q, grad_p, grad, Agrad, sigma, dW, h)
end