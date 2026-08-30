# Generate the integrator physical-oracle receipt from a clean ReactiveHMC
# checkout. The source checkout, input model, and every integration control are
# explicit so the candidate compiler cannot supply its own expectations.

using LinearAlgebra
using ReactiveHMC
using SHA

const EXPECTED_REACTIVEHMC_SHA =
    "ca9ea4ca41924bb0e1fadc01c717e1333916aba6"
const EXPECTED_INTEGRATORS_SHA256 =
    "39503d11f870d5942f9fe4a06065ea75d822b0702cc56c0824bff9f5d2c02b92"
const EXPECTED_PHASEPOINTS_SHA256 =
    "b2eb1d28c347412fafcf6e9e5cac6b4c6c08801e5b6e2db83826806a79bdaaba"

upstream_root = Base.pkgdir(ReactiveHMC)
upstream_sha = readchomp(`git -C $upstream_root rev-parse HEAD`)
upstream_sha == EXPECTED_REACTIVEHMC_SHA || error(
    "expected ReactiveHMC $EXPECTED_REACTIVEHMC_SHA, loaded $upstream_sha",
)
isempty(readchomp(`git -C $upstream_root status --short -- src`)) ||
    error("ReactiveHMC src/ checkout is dirty")

function require_digest(relative_path, expected)
    actual = bytes2hex(sha256(read(joinpath(upstream_root, relative_path))))
    actual == expected || error("expected $relative_path $expected, read $actual")
end
require_digest("src/integrators.jl", EXPECTED_INTEGRATORS_SHA256)
require_digest("src/phasepoints.jl", EXPECTED_PHASEPOINTS_SHA256)

potential(position) = sum(abs2, position) / 2
potential_gradient(position) = (potential(position), copy(position))
metric(position) = (potential_gradient(position)...,
                    Diagonal(1 .+ abs2.(position)))
function metric_gradient(position)
    pot, dpot, value = metric(position)
    derivative = zeros(eltype(position), length(position),
                       length(position), length(position))
    for index in eachindex(position)
        derivative[index, index, index] = 2position[index]
    end
    (pot, dpot, value, derivative)
end

const POSITION = [0.25, -0.5]
const MOMENTUM = [0.4, 0.1]

function new_euclidean()
    ReactiveHMC.euclidean_phasepoint(
        potential, potential_gradient, Diagonal(ones(2)),
        copy(POSITION), copy(MOMENTUM))
end
function new_riemannian()
    ReactiveHMC.riemannian_phasepoint(
        potential, potential_gradient, metric, metric_gradient,
        copy(POSITION), copy(MOMENTUM))
end
cases = NamedTuple[]

point = new_euclidean()
ReactiveHMC.leapfrog!(point; stepsize=0.1)
push!(cases, (name="leapfrog", stepsize=0.1, n_fi_steps=0, n_steps=1, point))

point = new_riemannian()
ReactiveHMC.generalized_leapfrog!(point; stepsize=0.1, n_fi_steps=4)
push!(cases, (name="generalized_leapfrog", stepsize=0.1,
              n_fi_steps=4, n_steps=1, point))

point = new_riemannian()
ReactiveHMC.implicit_midpoint!(point; stepsize=0.1, n_fi_steps=4)
push!(cases, (name="implicit_midpoint", stepsize=0.1,
              n_fi_steps=4, n_steps=1, point))

point = new_riemannian()
ReactiveHMC.multistep(
    ReactiveHMC.generalized_leapfrog!, point;
    stepsize=0.06, n_fi_steps=2, n_steps=3)
push!(cases, (name="multistep", stepsize=0.06,
              n_fi_steps=2, n_steps=3, point))

println("schema = \"reactivehmc-integrators-ca9-v1\"")
println()
println("[pins]")
println("reactivehmc_revision = \"$upstream_sha\"")
println("integrators_sha256 = \"$EXPECTED_INTEGRATORS_SHA256\"")
println("phasepoints_sha256 = \"$EXPECTED_PHASEPOINTS_SHA256\"")
println("julia_version = \"$(VERSION)\"")
println()
println("[inputs]")
println("position = $(repr(POSITION))")
println("momentum = $(repr(MOMENTUM))")

for case in cases
    result_point = case.point
    println()
    println("[[cases]]")
    println("name = \"$(case.name)\"")
    println("stepsize = $(repr(case.stepsize))")
    println("n_fi_steps = $(case.n_fi_steps)")
    println("n_steps = $(case.n_steps)")
    println("pos = $(repr(result_point.pos))")
    println("mom = $(repr(result_point.mom))")
    println("ham = $(repr(result_point.ham))")
    println("dham_dpos = $(repr(result_point.dham_dpos))")
    println("dham_dmom = $(repr(result_point.dham_dmom))")
end
