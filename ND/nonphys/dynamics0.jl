## -----------------------------------------------------------------------------
## efficient
## -----------------------------------------------------------------------------
function U!(C::Cache)
    ## p^T I p
    C.E.pe = 0.0
    @simd for i in eachindex(C.S.p)
        @inbounds @views C.E.pe += C.S.p[i]^2
    end
    C.E.pe *= 0.5

    ## 0.5 sin(q)^T B sin(q)
    C.E.ke = 0.0
    @simd for i in eachindex(C.S.q)
        @simd for j in eachindex(C.S.q)
            @inbounds @views C.E.ke += C.B[i,j] * sin(C.S.q[i]) * sin(C.S.q[j])
        end
    end
    C.E.ke *= 0.5
    C.E.te = C.E.pe + C.E.ke
    nothing
end

function dUdq!(C::Cache)
    ## B sin(q) ∘ cos(q)
    mul!(C.grad_q, C.B, sin.(C.S.q))
    @simd for i in eachindex(C.S.q)
        @inbounds @views C.grad_q[i] *= cos(C.S.q[i])
    end
    nothing
end

function dUdp!(C::Cache)
    ## I p
    @simd for i in eachindex(C.S.p)
        @inbounds @views C.grad_p[i] = C.S.p[i]
    end
    nothing
end

function grad!(C::Cache)
    dUdq!(C)
    dUdp!(C)
    ## grad = [dUdq; dUdp], NOTE that `C.A` does the "rearranging"
    @simd for i in eachindex(C.grad_q)
        @inbounds @views C.grad[i] = C.grad_q[i]
        @inbounds @views C.grad[i+C.n] = C.grad_p[i]
    end
    nothing
end
dU! = grad!

function noise!(C::Cache)
    randn!(C.dW)
    @simd for i in eachindex(C.dW)
        @inbounds @views C.dW[i] = C.sigma[i] * C.dW[i]
    end
    nothing
end

function feuler!(C::Cache)
    ## calc grad
    grad!(C)

    ## prefactor dynamics
    mul!(C.Agrad, C.A, C.grad)

    ## q: A_q * p + dW_q
    @simd for i in eachindex(C.S.q)
        @inbounds @views C.S.q[i] += C.h * C.Agrad[i] + C.sigma[i] * C.dW[i]
    end

    ## p: A_p * dUdq + dW_p
    @simd for i in eachindex(C.S.p)
        @inbounds @views C.S.p[i] += C.h * C.Agrad[i] + C.sigma[i] * C.dW[i]
    end
    nothing
end

function constraint!(C::Cache)
    """ !! assumes `C.S.L`'s links are sorted and ordered within C.CE!! """
    """ linear link `l = (i,j)`: b_{ij} |θ_i - θ_j| ≥ α * b_{ij} """
    @simd for i in eachindex(C.S.L)
        @inbounds @views C.CE.ce[i] = C.S.L[i].b * abs(C.S.q[C.S.L[i].head] - C.S.q[C.S.L[i].tail])
        @inbounds @views C.CE.failed[i] = C.CE.ce[i] >= C.CE.limit[i]
    end
end

"""
## `set_A!`: set DETERMINISTIC dynamics through `C.A` to include DAMPING on different dimensions
### arguments:
    * `C::Cache`
    * `parms::Dict`: parameter dictionary containing
        - [`parms[:gamma]::Float64`]: (uniform) damping value
        - [`parms[:q_damp]::Bool`]: add damping to `dq` dynamics (NON-STANDARD)
        - [`parms[:p_damp]::Bool`]: add damping to `dp` dynamics (STANDARD LANGEVIN)
"""
function set_A!(C::Cache, parms::Dict)
    @assert(haskey(parms, :gamma))
    @assert(haskey(parms, :q_damp))
    @assert(haskey(parms, :p_damp))
    if parms[:q_damp] == true
        @info "adding deterministic DAMPING to `dq` dynamics is NON-STANDARD"
        for i = 1:C.n
            C.A[i, i] = -copy(parms[:gamma])
        end
    end
    if parms[:p_damp] == true
        for i = 1:C.n
            C.A[C.n+i, C.n+i] = -copy(parms[:gamma])
        end
    end
    @info "S" -Diagonal(Matrix(C.A))
end

"""
## `set_sigma!`: set STOCHASTIC dynamics through `C.sigma` to include NOISE on different dimensions
### arguments:
    * `C::Cache`
    * `parms::Dict`: parameter dictionary containing
        - [`parms[:h]::Bool`]: step size
        - [`parms[:gamma]::Float64`]: damping value
        - [`parms[:tau]::Bool`]: temperature value
        - [`parms[:q_noise]::Bool`]: add noise to `dq` dynamics (NON-STANDARD)
        - [`parms[:p_noise]::Bool`]: add noise to `dp` dynamics (STANDARD LANGEVIN)
"""
function set_sigma!(C::Cache, parms::Dict)
    @assert(haskey(parms, :h))
    @assert(haskey(parms, :gamma))
    @assert(haskey(parms, :tau))
    sigma = sqrt(2.0 * parms[:h] * parms[:gamma] * parms[:tau])
    @debug "setting sigma; assumes `S = I`"
    if parms[:q_damp] == true
        @info "adding stochastic NOISE to `dq` dynamics is NON-STANDARD"
        for i = 1:C.n
            C.sigma[i] = copy(sigma)
        end
    end
    if parms[:p_damp] == true
        for i = 1:C.n
            C.sigma[C.n+i] = copy(sigma)
        end
    end
    @info "sigma" C.sigma
end
## -----------------------------------------------------------------------------


## -----------------------------------------------------------------------------
## convenient
## -----------------------------------------------------------------------------
function U(q, p, B::AbstractArray)
    return 0.5 * (p'*p + (sin.(q)' * B * sin.(q)))
end
function dU(q, p, B::AbstractArray)
    n = length(q); @assert(n == length(p))
    N = 2n
    x = [q; p]
    u = x -> (q = x[1:n]; p = x[(n+1):(N)]; U(q, p, B))
    return ForwardDiff.gradient(u, x)
end
function d2U(q, p, B::AbstractArray)
    n = length(q); @assert(n == length(p))
    N = 2n
    x = [q; p]
    u = x -> (q = x[1:n]; p = x[(n+1):(N)]; U(q, p, B))
    return ForwardDiff.hessian(u, [q; p])
end
function constraint(q, p, L::Link)
    return L.b * abs(q[L.head] - q[L.tail])
end
function dconstraint(q, p, L::Link)
    n = length(q); @assert(n == length(p))
    N = 2n
    x = [q; p]
    c = x -> (q = x[1:n]; p = x[(n+1):(N)]; constraint(q, p, B))
    return ForwardDiff.gradient(c, x)
end
function d2constraint(q, p, L::Link)
    n = length(q); @assert(n == length(p))
    N = 2n
    x = [q; p]
    c = x -> (q = x[1:n]; p = x[(n+1):(N)]; constraint(q, p, B))
    return ForwardDiff.hessian(c, [q; p])
end

## -----------------------------------------------------------------------------