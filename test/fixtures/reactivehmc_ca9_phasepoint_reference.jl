# Generate the six-variant phase-point physical-oracle receipt from a clean
# ReactiveHMC checkout. This process imports no ReactiveKernels code.

using LinearAlgebra
using ReactiveHMC
using SHA

const EXPECTED_REACTIVEHMC_SHA =
    "ca9ea4ca41924bb0e1fadc01c717e1333916aba6"
const EXPECTED_PHASEPOINTS_SHA256 =
    "b2eb1d28c347412fafcf6e9e5cac6b4c6c08801e5b6e2db83826806a79bdaaba"

upstream_root = Base.pkgdir(ReactiveHMC)
upstream_sha = readchomp(`git -C $upstream_root rev-parse HEAD`)
upstream_sha == EXPECTED_REACTIVEHMC_SHA || error(
    "expected ReactiveHMC $EXPECTED_REACTIVEHMC_SHA, loaded $upstream_sha",
)
isempty(readchomp(`git -C $upstream_root status --short -- src`)) ||
    error("ReactiveHMC src/ checkout is dirty")
phasepoints_sha256 = bytes2hex(sha256(read(joinpath(
    upstream_root, "src", "phasepoints.jl"))))
phasepoints_sha256 == EXPECTED_PHASEPOINTS_SHA256 || error(
    "expected phasepoints.jl $EXPECTED_PHASEPOINTS_SHA256, read $phasepoints_sha256",
)

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
const METRIC = Diagonal(ones(2))
const SPEED = 1.5
const MASS = 0.8
const ALPHA = 20.0

cases = (
    (name="euclidean", point=ReactiveHMC.euclidean_phasepoint(
        potential, potential_gradient, METRIC, copy(POSITION), copy(MOMENTUM))),
    (name="riemannian", point=ReactiveHMC.riemannian_phasepoint(
        potential, potential_gradient, metric, metric_gradient,
        copy(POSITION), copy(MOMENTUM))),
    (name="softabs", point=ReactiveHMC.riemannian_softabs_phasepoint(
        potential, potential_gradient, metric, metric_gradient,
        copy(POSITION), copy(MOMENTUM); alpha=ALPHA)),
    (name="relativistic_euclidean",
     point=ReactiveHMC.relativistic_euclidean_phasepoint(
        potential, potential_gradient, METRIC, copy(POSITION), copy(MOMENTUM);
        c=SPEED, m=MASS)),
    (name="relativistic_riemannian",
     point=ReactiveHMC.relativistic_riemannian_phasepoint(
        potential, potential_gradient, metric, metric_gradient,
        copy(POSITION), copy(MOMENTUM); c=SPEED, m=MASS)),
    (name="relativistic_softabs",
     point=ReactiveHMC.relativistic_riemannian_softabs_phasepoint(
        potential, potential_gradient, metric, metric_gradient,
        copy(POSITION), copy(MOMENTUM); alpha=ALPHA, c=SPEED, m=MASS)),
)

println("schema = \"reactivehmc-phasepoints-ca9-v1\"")
println()
println("[pins]")
println("reactivehmc_revision = \"$upstream_sha\"")
println("phasepoints_sha256 = \"$phasepoints_sha256\"")
println("julia_version = \"$(VERSION)\"")
println()
println("[inputs]")
println("position = $(repr(POSITION))")
println("momentum = $(repr(MOMENTUM))")
println("speed = $(repr(SPEED))")
println("mass = $(repr(MASS))")
println("alpha = $(repr(ALPHA))")

for case in cases
    point = case.point
    println()
    println("[[cases]]")
    println("name = \"$(case.name)\"")
    println("pot = $(repr(point.pot))")
    println("ham = $(repr(point.ham))")
    println("dham_dpos = $(repr(point.dham_dpos))")
    println("dham_dmom = $(repr(point.dham_dmom))")
end
