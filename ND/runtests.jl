using Test
include("mfpt.jl")

## -----------------------------------------------------------------------------
## input
## -----------------------------------------------------------------------------
D = 3
const tol = 1e-9
const links = [Link(1, 1, 2, 0.5); Link(2, 1, 3, 0.5); Link(3, 2, 3, 0.5)]
logger=Logging.ConsoleLogger(stderr, Logging.Debug)
parms = Dict()
parms[:h] = 0.01
parms[:q_damp] = false
parms[:p_damp] = true
parms[:gamma] = 1.0
parms[:tau] = 0.1
parms[:fail_kind] = 1
## -----------------------------------------------------------------------------

function load(D)
    n = D
    N = 2n
    xrand = randn(N)
    qrand = xrand[1:n]
    prand = xrand[(n+1):(N)]
    Yrand = randn(n, n)
    Yrand = Yrand' * Yrand
    evals = eigvals(Yrand)
    evecs = eigvecs(Yrand)
    Yrand = Yrand .- evals[1] * evecs[:,1]*evecs[:,1]'  ## one zero eig-val
    S = State(qrand, prand, links)
    C = Cache(S)
    C.Y = Yrand
    return S, C
end
S, C = load(D)
U!(C)
dU!(C)
constraint!(C)

em = energy_model(C, parms=parms)
solve(em)
d = setup(em)

bar = get_bar(C, parms=parms)
star = get_star(C, parms=parms)

cstar = [NaN]
c!(cstar, xstar, model=star[:dstar])

## -----------------------------------------------------------------------------
## tests
## -----------------------------------------------------------------------------
@testset "dynamics" begin
@test norm(S.q - C.S.q) <= tol
@test norm(S.p - C.S.p) <= tol
@test abs(U(C.S.q, C.S.p, C.Y) - C.E.te) <= tol
@test norm(dU(C.S.q, C.S.p, C.Y) - C.grad) <= tol
for i in eachindex(C.CE.e)
    @test abs(C.CE.e[i] - constraint(C.S.q, C.S.p, C.S.L[i])) <= tol
end
end

## using `Logging`
Logging.with_logger(logger) do; set_A!(C, parms); end
## -----------------------------------------------------------------------------