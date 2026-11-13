module CloudKernelSMCTests
# Cloud-fitted stage kernels (SMCOptions(cloud_kernels=J)) on a target whose two
# equally probable modes need opposite scales: mode A is narrow in x1, mode B in
# x2, as relabelled mixture components are (snag tempered-smc-on-dc64cd8b).
using ReactiveKernelsPPL, DifferentiationInterface, Enzyme, LogDensityProblems
using SequentialMonteCarlo, WarmupHMC, Random, Statistics, Test, Serialization
import LinearAlgebra

const EXT = Base.get_extension(ReactiveKernelsPPL, :ReactiveKernelsPPLSMCExt)
const BACKEND = AutoEnzyme(mode=Enzyme.Reverse)
const NARROW, WIDE, OFFSET, LOGMASS = 0.01, 1.0, 3.0, 0.4

struct SwappedModes
    gradient::Vector{Float64}
end
LogDensityProblems.capabilities(::Type{SwappedModes}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(::SwappedModes) = 2

# exp(LOGMASS) * (N(x; (OFFSET,0), diag(NARROW,WIDE)^2) + N(x; (0,OFFSET), diag(WIDE,NARROW)^2)) / 2
function components(u)
    a = -((u[1]-OFFSET)/NARROW)^2/2 - (u[2]/WIDE)^2/2 - log(NARROW*WIDE)
    b = -(u[1]/WIDE)^2/2 - ((u[2]-OFFSET)/NARROW)^2/2 - log(NARROW*WIDE)
    return a, b
end
function value(u)
    a, b = components(u)
    m = max(a, b)
    return LOGMASS + m + log(exp(a-m) + exp(b-m)) - log(2) - log(2pi)
end
LogDensityProblems.logdensity(::SwappedModes, u) = value(u)
function mixture_gradient(u)
    a, b = components(u)
    m = max(a, b)
    pa, pb = exp(a-m), exp(b-m)
    s = pa + pb
    pa, pb = pa/s, pb/s
    return [pa*(-(u[1]-OFFSET)/NARROW^2) + pb*(-u[1]/WIDE^2),
            pa*(-u[2]/WIDE^2) + pb*(-(u[2]-OFFSET)/NARROW^2)]
end
function LogDensityProblems.logdensity_and_gradient(p::SwappedModes, u)
    p.gradient .= mixture_gradient(u)
    return value(u), p.gradient
end

target() = prepare_smc(() -> SwappedModes(zeros(2)), zeros(2); backend=BACKEND,
    model_id="swapped-modes-v1", binding_id="none", frame_id="x1-x2")
reference() = SMCReference([1.5, 1.5], [3.0, 3.0])
mass_a(s) = sum(s.weights[j] for j in axes(s.positions, 2) if s.positions[1, j] > s.positions[2, j]; init=0.0)
divergent(s) = sum(h.divergences for h in s.history) / sum(h.moves for h in s.history)
snapshot(s) = (; s.positions, s.weights, s.temperature, s.log_normalizer,
    s.ancestors, s.lineage, s.history, rng=rand(copy(s.rng), UInt64, 8))

@testset "Cloud kernel options" begin
    @test SMCOptions().cloud_kernels == 0
    @test SMCOptions(cloud_kernels=3).cloud_kernels == 3
    @test_throws ArgumentError SMCOptions(cloud_kernels=-1)
    @test_throws ArgumentError SMCOptions(cloud_kernels=1.5)
    @test_throws ArgumentError tune_smc(target(); rng=Xoshiro(1), init=zeros(2),
        options=SMCOptions(cloud_kernels=2))
    # Without cloud kernels every stage records the fixed kernel.
    options = SMCOptions(; moves=2, max_tree_depth=5, stepsize=0.2, nthreads=1)
    plain = smc(target(); rng=Xoshiro(3), particles=64, reference=reference(), options)
    @test all(!haskey(h.kernel, :components) for h in plain.history)
end

@testset "Mixture fit separates modes with opposite scales" begin
    rng = Xoshiro(5)
    X = hcat([OFFSET, 0.0] .+ [NARROW, WIDE] .* randn(rng, 2, 300),
             [0.0, OFFSET] .+ [WIDE, NARROW] .* randn(rng, 2, 300))
    w = fill(1/600, 600)
    fit = EXT._cloud_fit(X, w, 4, Xoshiro(9))
    @test fit.components == 2
    order = sortperm(fit.mean[1, :]; rev=true)
    @test fit.sd[:, order[1]] ≈ [NARROW, WIDE] rtol=0.15
    @test fit.sd[:, order[2]] ≈ [WIDE, NARROW] rtol=0.15
    @test EXT._cloud_fit(X[:, 1:300], fill(1/300, 300), 4, Xoshiro(9)).components == 1
    # Ten effective particles per component at least: four points fit one.
    @test EXT._cloud_fit(X[:, [1, 2, 301, 302]], fill(0.25, 4), 4, Xoshiro(9)).components == 1
end

# Overlapping scale mixture 0.5 N(0, I) + 0.5 N(0, 0.05^2 I): the selected
# component depends smoothly on the position, so only the min(1, r_j(x')/r_j(x))
# correction keeps it invariant (without it, P(|x| < 0.1) drifts from 0.435 to 0.22).
struct ScaleMixture end
LogDensityProblems.capabilities(::Type{ScaleMixture}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(::ScaleMixture) = 2
function scale_parts(u)
    r2 = sum(abs2, u)
    return -r2/2, -r2/(2*0.05^2) - 2log(0.05)
end
function LogDensityProblems.logdensity(::ScaleMixture, u)
    a, b = scale_parts(u)
    m = max(a, b)
    return m + log(exp(a-m) + exp(b-m))
end
function LogDensityProblems.logdensity_and_gradient(p::ScaleMixture, u)
    a, b = scale_parts(u)
    m = max(a, b)
    pa, pb = exp(a-m), exp(b-m)
    s = pa + pb
    return m + log(s), -(pa/s) .* u .- (pb/s) .* u ./ 0.05^2
end

@testset "Cloud mixture moves keep an exact sample exact" begin
    rng = Xoshiro(1)
    n = 2000
    X = reduce(hcat, [(rand(rng) < 0.5 ? 1.0 : 0.05) .* randn(rng, 2) for _ in 1:n])
    w = fill(1/n, n)
    fit = EXT._cloud_fit(X, w, 4, rng)
    @test fit.components == 2
    kernel = (; kind=:cloud, components=fit.components, logweights=fit.logweight,
        means=fit.mean, scales=fit.sd, stepsizes=zeros(2), moves=20, max_tree_depth=7)
    r = zeros(2)
    R = reduce(hcat, [copy(EXT._cloud_responsibilities!(r, kernel, X[:, i])) for i in 1:n])
    for j in 1:2
        starts = [X[:, EXT._cloud_categorical(rng, w .* R[j, :])] for _ in 1:16]
        kernel.stepsizes[j] = first(EXT._cloud_stepsize(ScaleMixture(), starts,
            LinearAlgebra.Diagonal(fit.sd[:, j]), 0.25, 7, rng))
    end
    moved = similar(X)
    result = EXT._mutate_cloud!(moved, (; positions=X), collect(1:n), rand(rng, UInt64, n),
        ScaleMixture(), kernel, 1:n)
    near(Y) = mean(sqrt.(vec(sum(abs2, Y; dims=1))) .< 0.1)
    @test result.rejections > 0
    @test near(moved) ≈ (2 - exp(-0.005) - exp(-2))/2 atol=0.04  # exact 0.4348
    @test mean(vec(sum(abs2, moved; dims=1))) ≈ 1.0025 atol=0.12
end

@testset "Cloud kernels: both modes, evidence, replay" begin
    t = target()
    frozen = smc(t; rng=Xoshiro(31), particles=1024, reference=reference(),
        options=SMCOptions(; moves=4, max_tree_depth=7, stepsize=0.5, metric=[NARROW, WIDE], nthreads=1))
    cloud = smc(t; rng=Xoshiro(31), particles=1024, reference=reference(),
        options=SMCOptions(; moves=4, max_tree_depth=7, nthreads=1, cloud_kernels=4))
    # One fixed metric suited to mode A diverges in mode B.
    @test divergent(frozen) > 0.2
    @test divergent(cloud) < 0.02
    @test mass_a(cloud) ≈ 0.5 atol=0.08
    @test cloud.log_normalizer ≈ LOGMASS atol=0.15
    h = cloud.history[end]
    @test h.kernel.kind === :cloud
    @test 1 <= h.kernel.components <= 4
    @test length(h.kernel.stepsizes) == h.kernel.components
    @test all(>(0), h.kernel.stepsizes)
    @test h.kernel.tuning_evaluations > 0
    @test h.evaluations > h.kernel.tuning_evaluations
    @test h.kernel.selection_rejections >= 0
    @test any(x.kernel.components == 2 for x in cloud.history)

    options = SMCOptions(; moves=2, max_tree_depth=6, cloud_kernels=3, nthreads=1)
    start = initialize_smc(t; rng=Xoshiro(77), particles=97, reference=reference(), options)
    partial = smc(start; stages=3)
    serial = smc(partial)
    @test snapshot(serial) == snapshot(smc(resume_smc(t, smc_checkpoint(partial))))
    threaded = smc(initialize_smc(t; rng=Xoshiro(77), particles=97, reference=reference(),
        options=SMCOptions(; moves=2, max_tree_depth=6, cloud_kernels=3,
            nthreads=Threads.nthreads())))
    @test threaded.positions == serial.positions
    @test threaded.log_normalizer == serial.log_normalizer
    @test sum(x.evaluations for x in threaded.history) == sum(x.evaluations for x in serial.history)
end
end
