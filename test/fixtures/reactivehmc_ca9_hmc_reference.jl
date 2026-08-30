# Generate the fixed-step HMC physical-oracle receipt from a clean ReactiveHMC
# checkout.  Randomness is a declared draw script so consumption and early exit
# are exact observables rather than inferred from a seed.

using LinearAlgebra
using Random
using ReactiveHMC
using SHA

const EXPECTED_REACTIVEHMC_SHA =
    "ca9ea4ca41924bb0e1fadc01c717e1333916aba6"
const EXPECTED_HMC_SHA256 =
    "5d341facd929201ada08800e8d0194ec187f637ae036dd448461022a2bb577ea"

upstream_root = Base.pkgdir(ReactiveHMC)
upstream_sha = readchomp(`git -C $upstream_root rev-parse HEAD`)
upstream_sha == EXPECTED_REACTIVEHMC_SHA || error(
    "expected ReactiveHMC $EXPECTED_REACTIVEHMC_SHA, loaded $upstream_sha",
)
isempty(readchomp(`git -C $upstream_root status --short -- src`)) ||
    error("ReactiveHMC src/ checkout is dirty")

hmc_path = joinpath(upstream_root, "src", "hmc.jl")
hmc_sha256 = bytes2hex(sha256(read(hmc_path)))
hmc_sha256 == EXPECTED_HMC_SHA256 || error(
    "expected hmc.jl $EXPECTED_HMC_SHA256, read $hmc_sha256",
)

mutable struct ScriptedRNG <: AbstractRNG
    normal::Vector{Float64}
    exponential::Vector{Float64}
    normal_calls::Int
    exponential_calls::Int
    events::Vector{String}
end

ScriptedRNG(normal, exponential) =
    ScriptedRNG(copy(normal), copy(exponential), 0, 0, String[])

function Random.randn!(rng::ScriptedRNG, destination::AbstractVector)
    rng.normal_calls += 1
    push!(rng.events, "normal")
    copyto!(destination, rng.normal)
end

function Random.randexp(rng::ScriptedRNG)
    rng.exponential_calls += 1
    push!(rng.events, "exponential")
    rng.exponential_calls <= length(rng.exponential) ||
        error("unexpected exponential draw $(rng.exponential_calls)")
    rng.exponential[rng.exponential_calls]
end

potential(position) = sum(abs2, position) / 2
potential_gradient(position) = (potential(position), copy(position))

const CASES = (
    (name="accepted", stepsize=0.25, n_steps=3, min_dham=-1000.0,
     normal=[0.3, -0.4], exponential=[0.5]),
    (name="rejected", stepsize=1.25, n_steps=2, min_dham=-1000.0,
     normal=[1.2, -0.8], exponential=[0.01]),
    (name="diverged", stepsize=2.0, n_steps=3, min_dham=-0.1,
     normal=[1.2, -0.8], exponential=Float64[]),
)

println("schema = \"reactivehmc-hmc-ca9-v1\"")
println()
println("[pins]")
println("reactivehmc_revision = \"$upstream_sha\"")
println("hmc_sha256 = \"$hmc_sha256\"")
println("julia_version = \"$(VERSION)\"")

toml_array(values) = isempty(values) ? "[]" : repr(values)

for case in CASES
    initial_position = [0.1, -0.2]
    initial_momentum = zeros(2)
    point = ReactiveHMC.euclidean_phasepoint(
        potential,
        potential_gradient,
        Diagonal(ones(2)),
        copy(initial_position),
        copy(initial_momentum),
    )
    stepper = phasepoint -> ReactiveHMC.leapfrog!(phasepoint; stepsize=case.stepsize)
    energy_errors = Float64[]
    stats = state -> push!(energy_errors, state.dham)
    rng = ScriptedRNG(case.normal, case.exponential)
    state = ReactiveHMC.hmc_state(
        point;
        rng,
        n_steps=case.n_steps,
        min_dham=case.min_dham,
        step_f=stepper,
        stats_f=stats,
    )
    ReactiveHMC.step!(state)

    println()
    println("[[cases]]")
    println("name = \"$(case.name)\"")
    println("stepsize = $(repr(case.stepsize))")
    println("n_steps = $(case.n_steps)")
    println("min_dham = $(repr(case.min_dham))")
    println("initial_position = $(repr(initial_position))")
    println("normal_draw = $(repr(case.normal))")
    println("exponential_draws = $(toml_array(case.exponential))")
    println("normal_calls = $(rng.normal_calls)")
    println("exponential_calls = $(rng.exponential_calls)")
    println("rng_events = $(repr(rng.events))")
    println("energy_errors = $(repr(energy_errors))")
    println("init_pos = $(repr(state.init.pos))")
    println("init_mom = $(repr(state.init.mom))")
    println("init_ham = $(repr(state.init.ham))")
    println("fwd_pos = $(repr(state.fwd.pos))")
    println("fwd_mom = $(repr(state.fwd.mom))")
    println("fwd_ham = $(repr(state.fwd.ham))")
    println("dham = $(repr(state.dham))")
    println("diverged = $(state.diverged)")
end
