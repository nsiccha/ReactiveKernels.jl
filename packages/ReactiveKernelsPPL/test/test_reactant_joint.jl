# Reactant track: full-program `Reactant.@compile` of built RKPPL programs —
# primal lp parity and compiled value+gradient (Enzyme-through-Reactant)
# parity against the native kernels. Retained numerical plate/scan
# composition is covered separately by test_kernel_composition_reactant.jl.
#
# Call shape (the one thing that decides traceability): compile the RAW
# prepared kernel / `q.ad` directly (top level, or `invokelatest` AROUND
# `Reactant.compile` from an older world). A traced `invokelatest`
# WRAPPER is opaque to Reactant's overlay and RK's tensorized lowering, so
# the packed scalar reads `unconstrained[i]` die with `Scalar indexing is
# disallowed` — see the `prepare_query` docstring.
using DifferentiationInterface
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Random
using Test
import Distributions as RJDistributions
using LinearAlgebra: Cholesky

const _RJ_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

"""Central finite-difference gradient of a scalar kernel `f` at `u`."""
function _rj_findiff(f, u::Vector{Float64}; h = 1e-6)
    map(eachindex(u)) do i
        up = copy(u); um = copy(u)
        up[i] += h; um[i] -= h
        (f(up) - f(um)) / (2h)
    end
end

function _rj_scalar_offset_measure(built, bound, u)
    post = prepare_query(built, bound, :sampler)
    ru = Reactant.to_rarray(u)
    hlo = repr(Reactant.@code_hlo optimize = false post(ru))
    operations = Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith)\.\w+", hlo)
        operations[m.match] = get(operations, m.match, 0) + 1
    end
    @test !isempty(operations)
    native = post(u)
    compiled = Reactant.@compile post(ru)
    @test Float64(compiled(ru)) ≈ native rtol = 1e-9
    q = prepare_sampler(built, bound, u; backend = _RJ_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    @test val ≈ native rtol = 1e-12
    @test g ≈ _rj_findiff(post, u) rtol = 1e-5 atol = 1e-7
    cad = compile_ad_value_and_gradient(q.ad, ru)
    rval, rgrad = cad(ru)
    @test Float64(rval) ≈ native rtol = 1e-9
    @test Array(rgrad) ≈ g rtol = 1e-8 atol = 1e-9
    return operations
end

@testset "Reactant: positional scalar offsets retain broadcast structure" begin
    for kind in (:vector, :matrix)
        declaration, read = kind === :vector ?
            (:(z[1:3] .~ Normal.(0, 1)), :(z[2])) :
            (:(L ~ LKJCholesky(2, 1.0)), :(L[2, 1]))
        # Fixed scale isolates the offset from the known scale-gradient
        # compiler defect pinned by the tiny-model test below (§7n).
        plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            $declaration
            mu = a .- $read .+ x
            y .~ Normal.(mu, 0.7)
        end, (:y, :x); conditioned = (:y, :x))
        counts = Dict{String,Int}[]
        for n in (3, 8)
            x = collect(range(-0.8, 0.9; length = n))
            y = sin.(x)
            bound = bind_data(plan, Dict(:y => y, :x => x))
            built = build_kernel(bound)
            u = [0.37 * sin(1.3 * i) - 0.2 for i in 1:built.layout.total]
            push!(counts, Base.invokelatest(_rj_scalar_offset_measure, built, bound, u))
        end
        @test counts[1] == counts[2]
    end
end

function _rj_tiny_bound()
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a
            y .~ Normal.(mu, sigma)
        end, (:y,); conditioned = (:y,))
    bind_data(plan, Dict{Symbol,AbstractVector}(:y => [1.0, 2.0, 1.5]))
end

@testset "Reactant ladder 1: tiny model primal + gradient" begin
    bound = _rj_tiny_bound()
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.2, -0.1]
    native = post_q(u)
    @test native ≈ -9.392436144765 rtol = 1e-12
    # (a) direct compile of the raw prepared kernel
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    @test Float64(compiled(Reactant.to_rarray(u))) ≈ native rtol = 1e-9
    # (b) the world-age-safe spelling: barrier AROUND the compile
    compiled_wa = Base.invokelatest(Reactant.compile, post_q,
        (Reactant.to_rarray(u),))
    @test Float64(compiled_wa(Reactant.to_rarray(u))) ≈ native rtol = 1e-9
    # (c) compiled value + gradient over the sampler's prepared AD.
    # reactant-tiny-mo-88cd8d95: Reactant's DEFAULT pipeline silently
    # miscompiles this program's reverse — each lane's `-1` adjoint from
    # the per-lane `-log(σ)` is counted ONCE instead of n times (+(n-1)
    # on the log-σ coordinate; primal exact). The trace and Enzyme's
    # reverse are correct (`optimize = :only_enzyme` matches native to
    # 1e-15), so the correctness assertion pins that pipeline until
    # upstream fixes the pass (reactivekernels-use §7n; tracking issue
    # on nsiccha/ReactiveKernels.jl; `:no_slice_slice` does NOT help —
    # different pass than §7j). The default-pipeline `@test_broken`
    # below turns suite-red the day upstream fixes it — then drop the
    # pin and the marker together.
    q = prepare_sampler(built, bound, u; backend = _RJ_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    @test val ≈ native rtol = 1e-12
    @test g ≈ _rj_findiff(q, u) rtol = 1e-6
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u);
        optimize = :only_enzyme)
    rval, rgrad = cad(Reactant.to_rarray(u))
    @test Float64(rval) ≈ native rtol = 1e-9
    @test Array(rgrad) ≈ g rtol = 1e-9
    cad_default = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    _, rgrad_default = cad_default(Reactant.to_rarray(u))
    @test_broken Array(rgrad_default) ≈ g rtol = 1e-9
end

# A host Bool mask nested in a traced broadcast must retain dense array
# materialization. Explicit array priors keep the same gather and mask path.
@testset "Reactant ladder 1b: nested host mask primal + gradient" begin
    plan = lower_rkppl(quote
            a ~ Normal(0, 1)
            sd[1:2] .~ HalfNormal.(1)
            L ~ LKJCholesky(2, 1.0)
            z[levels(g), 1:2] .~ Normal.(0, 1)
            B = z * (sd .* L)'
            mu = a .+ B[g, 1] .+ B[g, 2] .* (c .== 2)
            y .~ Normal.(mu, 1.5)
        end, (:y, :c, :g); conditioned = (:y, :c, :g))
    data = Dict{Symbol,AbstractVector}(
        :y => [0.5, -1.2, 0.8, 1.5, -0.3, 0.9, -0.7, 1.1],
        :c => [1, 2, 2, 1, 2, 1, 2, 1],
        :g => [2, 1, 3, 1, 2, 3, 1, 2])
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = fill(0.1, built.layout.total)
    native = post_q(u)
    nt = constrain(built.layout, u)
    B = nt.z * (nt.sd .* nt.L)'
    mu = nt.a .+ B[data[:g], 1] .+ B[data[:g], 2] .* (data[:c] .== 2)
    normal = RJDistributions.Normal()
    half = RJDistributions.truncated(normal, 0, Inf)
    oracle = RJDistributions.logpdf(normal, nt.a) +
        sum(RJDistributions.logpdf.(half, nt.sd)) +
        sum(RJDistributions.logpdf.(normal, nt.z)) +
        RJDistributions.logpdf(RJDistributions.LKJCholesky(2, 1.0),
            Cholesky(nt.L, 'L', 0)) +
        sum(RJDistributions.logpdf.(RJDistributions.Normal.(mu, 1.5), data[:y])) +
        logjac(built.layout, u)
    @test native ≈ oracle rtol = 1e-12
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    @test Float64(compiled(Reactant.to_rarray(u))) ≈ native rtol = 1e-9
    q = prepare_sampler(built, bound, u; backend = _RJ_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    @test val ≈ native rtol = 1e-12
    @test g ≈ _rj_findiff(post_q, u) rtol = 1e-5 atol = 1e-7
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    @test Float64(rval) ≈ native rtol = 1e-9
    @test Array(rgrad) ≈ g rtol = 1e-9
end

# Multivariate slice priors (`mv_slices.jl`) under Reactant: simplex and
# ordered slices are whole-array broadcasts and reductions, so they
# compile with primal + compiled-gradient parity; multivariate normal
# slices run their forward substitution as a native row loop.
module RJSliceModels
colsums(B) = vec(sum(B; dims = 1))
end

function _rj_slices_bound(prog)
    plan = lower_rkppl(prog, (:y, :k); mod = RJSliceModels, conditioned = (:y, :k))
    return bind_data(plan, Dict{Symbol,AbstractVector}(
        :y => [0.3, -1.2, 2.1, 0.7, -0.4, 1.5, 0.2, -0.8],
        :k => [1, 2, 3, 1, 2, 3, 1, 2]))
end

function _rj_slices_measure(built, bound, post_q, u)
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _RJ_BACKEND)
    g = similar(u)
    sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; native, primal, g, rval = Float64(rval), rgrad = Array(rgrad))
end

function _rj_slices_check(prog)
    bound = _rj_slices_bound(prog)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.37 * sin(1.3 * i) - 0.2 for i in 1:built.layout.total]
    return Base.invokelatest(_rj_slices_measure, built, bound, post_q, u)
end

@testset "Reactant: slice priors" begin
    for prog in (
            :(begin
                s ~ Exponential(1)
                eachrow(P[levels(k), 1:3]) .~ Dirichlet([1.0, 2.0, 3.0])
                y .~ Normal.(P[k, 1] .+ P[k, 3], s)
            end),
            :(begin
                s ~ Exponential(1)
                A[levels(k), 1:3] .~ Exponential.(1)
                eachrow(P[levels(k), 1:3]) .~ Dirichlet.(eachrow(A))
                y .~ Normal.(P[k, 3], s)
            end),
            :(begin
                s ~ Exponential(1)
                m ~ Normal(0, 1)
                eachcol(C[1:3, levels(k)]) .~ Ordered(Normal(m, 2), 3)
                v = colsums(C)
                y .~ Normal.(v[k], s)
            end))
        r = _rj_slices_check(prog)
        @test r.primal ≈ r.native rtol = 1e-9
        @test r.rval ≈ r.native rtol = 1e-9
        @test r.rgrad ≈ r.g rtol = 1e-9
    end
    # Default preparation keeps transparent errors. The opt-in compiled
    # multivariate policy is covered separately below.
    @test_throws "native execution only" _rj_slices_check(:(begin
            s ~ Exponential(1)
            L ~ LKJCholesky(2, 2.0)
            sd[1:2] .~ Exponential.(1)
            F = sd .* L
            eachrow(B[levels(k), 1:2]) .~ MvNormalCholesky(zeros(2), F)
            y .~ Normal.(B[k, 1], s)
        end))
end

include("test_mv_solve_reactant.jl")
include("test_mv_orientations_reactant.jl")
