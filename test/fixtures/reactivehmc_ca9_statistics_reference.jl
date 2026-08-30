# Generate the trajectory/sampling-statistics physical-oracle receipt from a
# clean ReactiveHMC checkout. This process imports no ReactiveKernels code.

using ReactiveHMC
using SHA

const EXPECTED_REACTIVEHMC_SHA =
    "ca9ea4ca41924bb0e1fadc01c717e1333916aba6"
const EXPECTED_STATISTICS_SHA256 =
    "20baff1337a3e7c5926f01e104484168dd9783fe397366ebcb78ad3501eb1f69"

upstream_root = Base.pkgdir(ReactiveHMC)
upstream_sha = readchomp(`git -C $upstream_root rev-parse HEAD`)
upstream_sha == EXPECTED_REACTIVEHMC_SHA || error(
    "expected ReactiveHMC $EXPECTED_REACTIVEHMC_SHA, loaded $upstream_sha",
)
isempty(readchomp(`git -C $upstream_root status --short -- src`)) ||
    error("ReactiveHMC src/ checkout is dirty")
statistics_sha256 = bytes2hex(sha256(read(joinpath(
    upstream_root, "src", "statistics.jl"))))
statistics_sha256 == EXPECTED_STATISTICS_SHA256 || error(
    "expected statistics.jl $EXPECTED_STATISTICS_SHA256, read $statistics_sha256",
)

point(pos, dham_dpos, pot) = (; pos, dham_dpos, pot)
event(gofwd, pos, dham_dpos, pot, dham) =
    (; gofwd, fwd=point(pos, dham_dpos, pot), dham)
sample_state(pos, stepsize, diverged) =
    (; init=(; pos), step_f=(; stepsize), diverged)

const RESET = point([0.25, -0.5], [0.2, -0.4], 0.15625)
const EVENTS = (
    (name="append_one", state=event(
        true, [0.4, -0.3], [0.5, -0.6], 0.125, -0.1)),
    (name="prepend", state=event(
        false, [-0.2, 0.6], [-0.3, 0.7], 0.2, -0.2)),
    (name="append_two", state=event(
        true, [0.55, -0.1], [0.65, -0.2], 0.15625, -0.05)),
)
const SECOND_RESET = point([0.7, 0.8], [0.75, 0.85], 0.565)
const SECOND_EVENT = event(
    true, [0.9, 1.0], [0.95, 1.05], 0.905, -1.5)
const SAMPLES = (
    (name="first", state=sample_state([1.0, -1.0], 0.125, false)),
    (name="second", state=sample_state([-1.0, 2.0], 0.0625, true)),
)

columns(matrix) = [collect(column) for column in eachcol(matrix)]
toml_bools(values) = "[" * join(string.(values), ", ") * "]"

trajectory = ReactiveHMC.trajectory_stats(2)
ReactiveHMC.reset!(trajectory, RESET)
for item in EVENTS
    trajectory(item.state)
end

first_positions = columns(trajectory.positions)
first_gradients = columns(trajectory.gradients)
first_dhams = copy(trajectory.dhams)
first_pots = copy(trajectory.pots)
first_idxs = copy(trajectory.idxs)

sampling = ReactiveHMC.sampling_stats(trajectory)
sampling(SAMPLES[1].state, nothing)

# Mutating the trajectory after the first sampling callback proves that
# full_history/full_idxs are snapshots rather than aliases of live buffers.
ReactiveHMC.reset!(trajectory, SECOND_RESET)
trajectory(SECOND_EVENT)
sampling(SAMPLES[2].state, nothing)

println("schema = \"reactivehmc-statistics-ca9-v1\"")
println()
println("[pins]")
println("reactivehmc_revision = \"$upstream_sha\"")
println("statistics_sha256 = \"$statistics_sha256\"")
println("julia_version = \"$(VERSION)\"")
println()
println("[inputs]")
println("dimension = 2")
println("reset_pos = $(repr(RESET.pos))")
println("reset_dham_dpos = $(repr(RESET.dham_dpos))")
println("reset_pot = $(repr(RESET.pot))")

for item in EVENTS
    state = item.state
    println()
    println("[[events]]")
    println("name = \"$(item.name)\"")
    println("go_forward = $(state.gofwd)")
    println("pos = $(repr(state.fwd.pos))")
    println("dham_dpos = $(repr(state.fwd.dham_dpos))")
    println("pot = $(repr(state.fwd.pot))")
    println("dham = $(repr(state.dham))")
end

println()
println("[second_trajectory]")
println("reset_pos = $(repr(SECOND_RESET.pos))")
println("reset_dham_dpos = $(repr(SECOND_RESET.dham_dpos))")
println("reset_pot = $(repr(SECOND_RESET.pot))")
println("go_forward = $(SECOND_EVENT.gofwd)")
println("pos = $(repr(SECOND_EVENT.fwd.pos))")
println("dham_dpos = $(repr(SECOND_EVENT.fwd.dham_dpos))")
println("pot = $(repr(SECOND_EVENT.fwd.pot))")
println("dham = $(repr(SECOND_EVENT.dham))")

println()
println("[trajectory]")
println("positions = $(repr(first_positions))")
println("gradients = $(repr(first_gradients))")
println("dhams = $(repr(first_dhams))")
println("pots = $(repr(first_pots))")
println("idxs = $(repr(first_idxs))")

for item in SAMPLES
    state = item.state
    println()
    println("[[samples]]")
    println("name = \"$(item.name)\"")
    println("init_pos = $(repr(state.init.pos))")
    println("stepsize = $(repr(state.step_f.stepsize))")
    println("diverged = $(state.diverged)")
end

println()
println("[sampling]")
println("draws = $(repr(columns(sampling.draws)))")
println("n_steps = $(repr(sampling.n_steps))")
println("stepsizes = $(repr(sampling.stepsizes))")
println("acc_rate = $(repr(sampling.acc_rate))")
println("diverged = $(toml_bools(sampling.diverged))")
println("full_history = $(repr(map(columns, sampling.full_history)))")
println("full_idxs = $(repr(sampling.full_idxs))")
