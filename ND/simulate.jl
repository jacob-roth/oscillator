"""
## `simulate_fail`: simulate model until failure
### arguments:
    * `C::Cache`
    * `parms::Dict`: parameter dictionary containing
        - [`parms[:tau]::Float64`]: temperature value
        - [`parms[:gamma]::Float64`]: (uniform) damping value
        - [`parms[:q_damp]::Bool`]: add damping to `dq` dynamics (NON-STANDARD)
        - [`parms[:p_damp]::Bool`]: add damping to `dp` dynamics (STANDARD LANGEVIN)
        - [`parms[:loglevel]::Int64`]: level of logging
        - [`parms[:fail_kind]::Int64`]: `0`-any failure, `k`-specific failure (`k ≠ 0`)
        - [`parms[:verb]::Bool`]: verbose
"""
function simulate_fail(S::State, parms::Dict)
    ## setup from input
    C = Cache(S)
    max_steps = parms[:max_steps]
    C.h = parms[:h]
    Random.seed!(parms[:seed])
    fail_kind = parms[:fail_kind]
    limit = [C.S.L[i].limit for i in eachindex(C.S.L)]
    haskey(parms, :verb) ? (verb = parms[:verb]) : (verb = false)

    ## fail setup
    i::Int64 = 0
    fail::Int64 = 0
    fail_ID::Int64 = 0
    fail_idx::Int64 = 0
    newfails = zeros(Bool, C.NL)

    ## dynamics
    set_A!(C, parms)
    set_sigma!(C, parms)

    ## simulate
    while fail == 0
        ## indexing
        i += 1
        if i > max_steps
            ## no failures
            fail_idx = -1
            fail_ID = -1
            break
        end

        ## update
        #### noise
        noise!(C)
        #### forward euler step
        feuler!(C)
        #### energy
        U!(C)
        #### constraint
        constraint!(C)

        ## check constraint
        if fail_kind == 0
            ## any failure
            @inbounds @views newfails .= (C.CE.e .>= limit)
            if any(newfails)
                ## update
                fail += 1
                fail_idx = i+1
                fail_ID = collect(1:C.NL)[newfails]
                ## display
                if verb
                    println("    !! (any) constraint failure !!")
                    println("    -> idx : $(fail_idx)")
                    println("    -> IDs : $(fail_ID)\n")
                end
            end
        else
            ## specific  failure
            if C.les[lf_kind] >= limit[lf_kind]
                fail += 1
                fail_idx = i+1
                fail_ID = lf_kind
                ## display
                if verb
                    println("    !! (specific) constraint failure !!")
                    println("    -> idx : $(fail_idx)")
                    println("    -> IDs : $(fail_ID)\n")
                end
            end
        end
    end

    ## return
    out = Dict()
    out[:x] = [C.S.q; C.S.p]
    out[:ec] = C.EC.e
    out[:fail_idx] = fail_idx
    out[:fail_ID] = fail_ID
    return out
end