# Generate the fixed-reference NUTS receipt from the upstream implementation.
# Run this file with ReactiveHMC.jl checked out at the exact revision below:
#
#   git clone https://github.com/nsiccha/ReactiveHMC.jl /tmp/ReactiveHMC-reference
#   git -C /tmp/ReactiveHMC-reference checkout ca9ea4ca41924bb0e1fadc01c717e1333916aba6
#   julia --startup-file=no --project=/tmp/ReactiveHMC-reference -e \
#       'using Pkg; Pkg.instantiate()'
#   julia --startup-file=no --project=/tmp/ReactiveHMC-reference \
#       test/fixtures/reactivehmc_ca9_reference.jl

using LinearAlgebra
using Random
using ReactiveHMC

const EXPECTED_REACTIVEHMC_SHA =
    "ca9ea4ca41924bb0e1fadc01c717e1333916aba6"

upstream_root = Base.pkgdir(ReactiveHMC)
upstream_sha = readchomp(`git -C $upstream_root rev-parse HEAD`)
upstream_sha == EXPECTED_REACTIVEHMC_SHA || error(
    "expected ReactiveHMC $EXPECTED_REACTIVEHMC_SHA, loaded $upstream_sha",
)

potential(position) = sum(abs2, position) / 2
potential_gradient(position) = (potential(position), copy(position))
point = euclidean_phasepoint(
    potential,
    potential_gradient,
    Diagonal(ones(2)),
    [0.1, -0.2],
    [0.3, -0.4],
)
stepper = phasepoint -> leapfrog!(phasepoint; stepsize = 0.25)
energy_errors = Float64[]
stats = state -> push!(energy_errors, state.dham)
state = nuts_state(
    point;
    rng = Xoshiro(42),
    step_f = stepper,
    stats_f = stats,
    max_depth = 3,
)
ReactiveHMC.step!(state)

n_steps = length(energy_errors)
receipt = (;
    pos = state.init.pos,
    mom = state.init.mom,
    ham = state.init.ham,
    depth = ceil(Int, log2(n_steps + 1)),
    n_steps,
    energy_error = state.dham,
    acceptance_rate = sum(min(1, exp(error)) for error in energy_errors) /
                      n_steps,
    diverged = state.diverged,
)
println(receipt)
