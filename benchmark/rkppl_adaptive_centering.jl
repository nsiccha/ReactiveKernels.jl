# Run in a consumer environment containing the optional centering-test
# dependencies. Reports warmed gradient time and allocation as N increases;
# the focused suite also checks that emitted body structure stays bounded.
using ReactiveKernels, ReactiveKernelsPPL, WarmupHMC, Enzyme, Test, Random, Statistics
using DifferentiationInterface: AutoEnzyme
import LogDensityProblems as LDP

include(joinpath(pkgdir(ReactiveKernelsPPL), "test", "centering", "runtests.jl"))
const ACT = AdaptiveCenteringTests

# Execute the displayed guide example without maintaining a second copy.
guide = read(joinpath(pkgdir(ReactiveKernels), "docs", "src", "rkppl.md"), String)
section = split(guide, "## Adaptive partial centering"; limit = 2)[2]
example = match(r"```julia\n(.*?)\n```"s, section).captures[1]
include_string(@__MODULE__, example, "rkppl.md:adaptive-centering")
@test LDP.logdensity(adaptive, u0) == LDP.logdensity(problem, u0)
println((; phase = :guide_example, passed = true))
flush(stdout)

function measurements(f, x; repetitions = 30)
    f(x)
    bytes = @allocated f(x)
    timings = [@elapsed(for _ in 1:repetitions; f(x); end) for _ in 1:3]
    (; seconds = minimum(timings) / repetitions, bytes)
end

function scaling()
    cells = (z = (location = :z_location, log_scale = :log_scale, source = 0.7),)
    built = nothing
    for n in (16, 64, 256, 1024)
        s = ACT.setup(n; built)
        built = s.built
        rp = adaptive_centering_problem(s.built, s.plan, s.problem, AutoEnzyme(); cells, u0 = s.u)
        base = measurements(x -> LDP.logdensity_and_gradient(s.problem, x), s.u)
        affine = measurements(x -> LDP.logdensity_and_gradient(rp, x), s.u)
        println((; phase = :scaling, n, base, affine,
            ratio = affine.seconds / base.seconds))
        flush(stdout)
    end
end

scaling()
