function energy_model(C::Cache; parms::Dict, print_level=4)
    # ## feasibility
    # if parms[:fail_kind] != 0
    #     ## feasibility model
    #     fm = Model(solver=IpoptSolver(print_level=print_level))
    #     # @variable(fm, -pi <= q[1:C.n] <= pi)
    #     @variable(fm, q[1:C.n])
    #     link = filter(x -> x.ID == parms[:fail_kind], C.S.L)[1]
    #     ## constraint energy
    #     @NLexpression(fm, c_l, (sin( q[link.head] - q[link.tail] )))
    #     ## link-limit constraint
    #     @NLconstraint(fm, ll, c_l >= link.limit)
    # end

    ## initialize
    em = Model(solver=IpoptSolver(print_level=print_level))
    ## variables
    @variable(em, -pi <= q[1:C.n] <= pi)  ## constrained
    @variable(em, p[1:C.n])               ## UNconstrained
    ## objective
    @NLexpression(em, pe, sum(p[i]^2 for i in eachindex(p)))
    @NLexpression(em, ke, sum(C.B[i, j] *
                                (  sin(q[i]) * sin(q[j])  )
                          for i in eachindex(q) for j in eachindex(q)))
    @NLobjective(em, Min, 0.5 * (ke + pe))

    ## don't add a duplicate constraint (deal w renaming later)
    @assert(:ll ∉ keys(em.objDict))

    ## constraint
    if parms[:fail_kind] != 0
        @info "ADDING specific constraint-$(parms[:fail_kind]) to energy model"
        link = filter(x -> x.ID == parms[:fail_kind], C.S.L)[1]
        limit = C.CE.limit[parms[:fail_kind]]
        ## constraint energy
        @NLexpression(em, c_l, link.b * (abs( q[link.head] - q[link.tail] )))
        ## link-limit constraint
        @NLconstraint(em, ll, c_l == limit)
    else
        @info "NOT adding a constraint to energy model"
    end
    return em
end
