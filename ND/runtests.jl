using Test
using BenchmarkTools
## -----------------------------------------------------------------------------
## input
## -----------------------------------------------------------------------------
const sim_kind = "dc" ## "hivolt", "nonphys", "ac", "dc"
include("mfpt.jl")
const D = 3
const tol = 1e-9
##                  ID  head  tail  alpha  susceptance
const links = [Link(1,  1,    2,    0.5,   abs(randn()));
               Link(2,  1,    3,    0.5,   abs(randn()));
               Link(3,  2,    3,    0.5,   abs(randn()))]
Pg = [1.0; 2.0; 0.0]
Pd = [0.0; 0.0; 3.0]
Pnet = Pg - Pd
logger=Logging.ConsoleLogger(stderr, Logging.Debug)
parms = Dict()
parms[:h] = 0.01
parms[:q_damp] = false
parms[:p_damp] = true
parms[:gamma] = 1.0
parms[:tau] = 0.1
parms[:fail_kind] = 1
## -----------------------------------------------------------------------------

function load_rand(D, links)
    n = D
    N = 2n
    xrand = randn(N)
    qrand = xrand[1:n]
    qrand[1] = 0.0
    prand = xrand[(n+1):(N)]
    S = State(qrand, prand, links, 1, [2;3])
    C = Cache(S)
    C.P = Pnet
    return S, C
end
S, C = load_rand(D, links)
U!(C)
dU!(C)
constraint!(C)

em = energy_model(C, parms=parms)
solve(em)
d = setup(em);

bar = get_bar(C, parms=parms)
star = get_star(C, parms=parms)

C.B[C.S.actID, C.S.actID] * bar[:S].q[C.S.actID]
Cbar = Cache(bar[:S])

f!(bar[:x], model=bar[:d])
f!(star[:x], model=star[:d])
cstar = [NaN]
c!(cstar, star[:x], model=star[:d])

## -----------------------------------------------------------------------------
## tests
## -----------------------------------------------------------------------------
@testset "dynamics" begin
@test norm(S.q - C.S.q) <= tol
@test norm(S.p - C.S.p) <= tol
@test abs(U(C.S.q, C.S.p, C.B, C.P, C.S.refID) - C.E.te) <= tol
@test norm(dU(C.S.q, C.S.p, C.B, C.P, C.S.refID) - C.grad) <= tol
for i in eachindex(C.CE.ce)
    @test abs(C.CE.ce[i] - constraint(C.S.q, C.S.p, C.S.L[i])) <= tol
end
end

@testset "structure" begin
for i = 1:100
    eigvals(random_Y(n))

## using `Logging`
Logging.with_logger(logger) do; set_A!(C, parms); end
## -----------------------------------------------------------------------------
