## -----------------------------------------------------------------------------
## system
## -----------------------------------------------------------------------------
function get_C(S::State)
    n = length(S.q); @assert(n == length(S.p))
    NL = length(S.L)
    CC = spzeros(NL, n)
    for l in S.L
        if (l.head != 0) && (l.tail != 0)
            CC[l.ID, l.head] = 1.0
            CC[l.ID, l.tail] = -1.0
        end
    end
    CC = Int64.(CC)
    return CC
end

function get_B(S::State)
    C = get_C(S)
    b = vec([l.b for l in S.L])
    B = C' * spdiagm(0 => b) * C
    return B
end
## -----------------------------------------------------------------------------


## -----------------------------------------------------------------------------
## optimization
## -----------------------------------------------------------------------------
function get_x(em::JuMP.Model)
    d = setup(em)
    return copy(d.last_x)
end

function get_x(d::JuMP.NLPEvaluator)
    return copy(d.last_x)
end

function update_state!(S::State, x::Array{Float64,1})
    N = length(x); @assert(mod(N/2, 2) == 0.0)
    n = div(N, 2)
    for i in S.actID
        S.q[i] = x[i]
    end
    for i in eachindex(S.p)
        S.p[i] = x[i+n-1]
    end
end

function get_bar(C::Cache; parms::Dict)
    p = deepcopy(parms); p[:fail_kind] = 0
    embar = energy_model(C, parms=p, print_level=0)
    solve(embar)
    dbar = setup(embar)
    xbar = get_x(embar)
    qbar = [C.S.q[C.S.refID]; xbar[1:(C.n-1)]]
    pbar = xbar[(C.n):(end)]
    Sbar = State(qbar, pbar, C.S.L, C.S.refID, C.S.actID)
    out = Dict()
    out[:d] = dbar
    out[:x] = xbar
    out[:S] = Sbar
    return out
end

function get_star(C::Cache; parms::Dict)
    p = deepcopy(parms)
    emstar = energy_model(C, parms=p, print_level=0)
    solve(emstar)
    dstar = setup(emstar)
    xstar = get_x(dstar)
    qstar = [C.S.q[C.S.refID]; xstar[1:(C.n-1)]]
    pstar = xstar[(C.n):(end)]
    Sstar = State(qstar, pstar, C.S.L, C.S.refID, C.S.actID)
    kstar = getdual(emstar[:ll])
    out = Dict()
    out[:d] = dstar
    out[:x] = xstar
    out[:S] = Sstar
    out[:k] = kstar
    return out
end
## -----------------------------------------------------------------------------