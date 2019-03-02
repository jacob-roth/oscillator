function energy_model(C::Cache; parms::Dict, print_level=4)
    if fail_kind != 0
        ## feasibility model
        fm = Model(solver=IpoptSolver(print_level=print_level))
        @variable(fm, -pi <= q[1:C.n] <= pi)
        link = filter(x -> x.ID == fail_kind, C.S.L)[1]
        ## constraint energy
        @NLexpression(fm, c_l, (sin( q[link.head] - q[link.tail] ))^2)
        ## link-limit constraint
        @NLconstraint(fm, ll, c_l >= link.limit)
    end

    em = Model(solver=IpoptSolver(print_level=print_level))
    ## variables
    @variable(em, -pi <= q[1:C.n] <= pi)  ## constrained
    @variable(em, p[1:C.n])               ## UNconstrained
    ## objective
    @NLexpression(em, pe, sum(p[i]^2 for i in eachindex(p)))
    @NLexpression(em, ke, sum(C.Y[i, j] *
                                (  sin(q[i]) * sin(q[j])  )
                          for i in eachindex(q) for j in eachindex(q)))
    @NLobjective(em, Min, 0.5 * (ke + pe))

    ## don't add a duplicate constraint (deal w renaming later)
    @assert(:ll ∉ keys(em.objDict))

    ## constraint
    fail_kind = parms[:fail_kind]
    if fail_kind != 0
        @info "ADDING specific constraint-$(fail_kind) to energy model"
        link = filter(x -> x.ID == fail_kind, C.S.L)[1]
        ## constraint energy
        @NLexpression(em, c_l, (sin( q[link.head] - q[link.tail] )))
        ## link-limit constraint
        @NLconstraint(em, ll, c_l >= link.limit)
    else
        @info "NOT adding a constraint to energy model"
    end
    return em
end

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
    for i in eachindex(S.q)
        S.q[i] = x[i]
        S.p[i] = x[i+n]
    end
end

function get_bar(C::Cache; parms::Dict)
    p = deepcopy(parms); p[:fail_kind] = 0
    embar = energy_model(C, parms=p, print_level=0)
    solve(embar)
    dbar = setup(embar)
    xbar = get_x(embar)
    Sbar = State(xbar[1:C.n], xbar[(C.n+1):(C.N)], C.S.L)
    out = Dict()
    out[:dbar] = dbar
    out[:xbar] = xbar
    out[:Sbar] = Sbar
    return out
end

function get_star(C::Cache; parms::Dict)
    p = deepcopy(parms)
    emstar = energy_model(C, parms=p, print_level=0)
    solve(emstar)
    dstar = setup(emstar)
    xstar = get_x(dstar)
    Sstar = State(xstar[1:C.n], xstar[(C.n+1):(C.N)], C.S.L)
    kstar = getdual(emstar[:ll])
    out = Dict()
    out[:dstar] = dstar
    out[:xstar] = xstar
    out[:Sstar] = Sstar
    out[:kstar] = kstar
    return out
end