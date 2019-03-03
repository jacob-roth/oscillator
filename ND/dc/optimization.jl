function energy_model(C::Cache; parms::Dict, print_level=4)
    ## initialize
    em = Model(solver=IpoptSolver(print_level=print_level))
    ## variables
    # @variable(em, -pi <= q[C.S.actID] <= pi)  ## constrained
    @variable(em, q[C.S.actID])  ## constrained
    @variable(em, p[1:C.n])                   ## UNconstrained
    qq = Array{Union{Float64, JuMP.Variable}}(undef, C.n)
    qq[C.S.refID] = 0.0
    for i in C.S.actID
        qq[i] = q[i]
    end
    ## objective
    @NLexpression(em, pe, sum(p[i]^2 for i in eachindex(p)))
    @NLexpression(em, ke, sum(C.B[id, jd] * qq[id] * qq[jd]
                          for id in C.S.actID for jd in C.S.actID))
    @NLexpression(em, de, sum(-C.P[id] * q[id] for id in C.S.actID))
    @NLobjective(em, Min, 0.5 * (ke + pe) + de)

    ## don't add a duplicate constraint (deal w renaming later)
    @assert(:ll ∉ keys(em.objDict))

    ## constraint
    if parms[:fail_kind] != 0
        @info "ADDING specific constraint-$(parms[:fail_kind]) to energy model"
        link = filter(x -> x.ID == parms[:fail_kind], C.S.L)[1]
        limit = C.CE.limit[parms[:fail_kind]]
        ## constraint energy
        @NLexpression(em, c_l, link.b * (abs( qq[link.head] - qq[link.tail] )))
        ## link-limit constraint
        @NLconstraint(em, ll, c_l == limit)
    else
        @info "NOT adding a constraint to energy model"
    end
    return em
end
