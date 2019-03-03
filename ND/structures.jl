mutable struct Link
    ID::Int64       ## dimension ID
    head::Int64     ## originates from this ID
    tail::Int64     ## ends at this ID
    alpha::Float64  ## constraint limit percentage of `b`
    b::Float64      ## link strength "susceptance"
end

mutable struct State
    ## TODO: how to add magnitude & loads
    q::Array{Float64,1}    ## like angle
    p::Array{Float64,1}    ## like frequency
    L::Array{Link,1}       ## like tline branches
    refID::Int64           ## reference node
    actID::Array{Int64,1}  ## active node
end
function State(q, p, L::Array{Link,1}, refID, actID)
    lIDs = [l.ID for l in L]
    @assert(sort(lIDs) == lIDs)
    return State(q, p, L)
end

mutable struct Energy
    pe::Float64  ## potential energy
    ke::Float64  ## kinetic energy
    de::Float64  ## dispatch energy
    te::Float64  ## total energy
end

mutable struct ConEnergy
    ce::Array{Float64,1}     ## constraint energy
    limit::Array{Float64,1}  ## constraint limit
    failed::Array{Bool,1}    ## link-constraint failure indicator
end

mutable struct Cache
    S::State
    n::Int64                  ## dimension `q` and `p`
    N::Int64                  ## system dimension `q` + `p`
    NL::Int64                 ## system link dimension
    P::Array{Float64,1}       ## net power `Pg - Pd`
    E::Energy                 ## hamiltonian energy
    CE::ConEnergy             ## constraint energy
    A::SparseMatrixCSC{Float64,Int64}  ## prefactor dynamics
    B::SparseMatrixCSC{Float64,Int64}  ## "susceptance" psd with `1` ∈ kernel
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
    P = zeros(n)
    E = Energy(0.0, 0.0, 0.0, 0.0)
    limit = [S.L[i].alpha * S.L[i].b for i in eachindex(S.L)]
    failed = [false for i in eachindex(S.L)]
    CE = ConEnergy(zeros(n), limit, failed)
    A = [spzeros(n, n)           spdiagm(0 => ones(n));
         spdiagm(0 => -ones(n))          spzeros(n, n)]
    B = get_B(S)
    grad_q = zeros(n)
    grad_p = zeros(n)
    grad = zeros(N)
    Agrad = zeros(N)
    sigma = zeros(N)
    dW = zeros(N)
    h = 0.0
    return Cache(S, n, N, NL, P, E, CE, A, B, grad_q, grad_p, grad, Agrad, sigma, dW, h)
end
function Cache(S::State, P::Array{Float64,1})
    n = length(S.q); @assert(n == length(S.p)); @assert(length(P) == n)
    N = 2n
    NL = length(S.L)
    E = Energy(0.0, 0.0, 0.0, 0.0)
    limit = [S.L[i].alpha * S.L[i].b for i in eachindex(S.L)]
    failed = [false for i in eachindex(S.L)]
    CE = ConEnergy(zeros(n), limit, failed)
    A = [spzeros(n, n)           spdiagm(0 => ones(n));
         spdiagm(0 => -ones(n))          spzeros(n, n)]
    B = get_B(S)
    grad_q = zeros(n)
    grad_p = zeros(n)
    grad = zeros(N)
    Agrad = zeros(N)
    sigma = zeros(N)
    dW = zeros(N)
    h = 0.0
    return Cache(S, n, N, NL, P, E, CE, A, B, grad_q, grad_p, grad, Agrad, sigma, dW, h)
end