# Reactant track: full-program `Reactant.@compile` of built RKPPL programs —
# primal lp parity and compiled value+gradient (Enzyme-through-Reactant)
# parity against the native kernels. The default grouped linear-PK path
# rejects compilation explicitly: unrolling its bound schedule violates the
# core constraints. The tiny model still proves the compiled AD path.
#
# Call shape (the one thing that decides traceability): compile the RAW
# prepared kernel / `q.ad` directly (top level, or `invokelatest` AROUND
# `Reactant.compile` from an older world). A traced `invokelatest`
# WRAPPER is opaque to Reactant's overlay, so `sum(view(unconstrained,
# i:i))` falls through to Base's scalar `mapreduce` and dies with
# `Scalar indexing is disallowed` — see the `prepare_query` docstring.
using DifferentiationInterface
using Enzyme
using ReactiveKernels
using ReactiveKernelsPPL
using Reactant
using Random
using Test

const _RJ_BACKEND = AutoEnzyme(; mode = Enzyme.Reverse)

"""Central finite-difference gradient of a scalar kernel `f` at `u`."""
function _rj_findiff(f, u::Vector{Float64}; h = 1e-6)
    map(eachindex(u)) do i
        up = copy(u); um = copy(u)
        up[i] += h; um[i] -= h
        (f(up) - f(um)) / (2h)
    end
end

function _rj_tiny_bound()
    plan = lower_rkppl(quote
            a ~ Normal(0, 5)
            sigma ~ Exponential(1)
            mu = a
            y .~ Normal.(mu, sigma)
        end, (:y,))
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

# Corpus 17 (varying dummy): the `(c .== level)` mask nests inside the LP's
# traced `.+`/`.*` fusion with no direct traced operand at the outermost
# call, so the lazy nest kept its host style and Reactant's per-argument
# standalone materialize produced the overlay-hostile `BitArray`
# (`StackOverflowError`; snag `dummy-varying-xl-3b05117e`). The lowering now
# dense-materializes host-only `Bool` nests (`Array{Bool}`), so the program
# compiles with primal + gradient parity like its ranef siblings.
@testset "Reactant ladder 1b: varying dummy (corpus 17) primal + gradient" begin
    plan = lower_rkppl(quote
            a ~ Normal(0, 1)
            r ~ varying_effect(g, [1, dummy(c, 2)])
            mu = a .+ r
            y .~ Normal.(mu, 1.5)
        end, (:y, :c, :g))
    bound = bind_data(plan, Dict{Symbol,AbstractVector}(
        :y => [0.5, -1.2, 0.8, 1.5, -0.3, 0.9, -0.7, 1.1],
        :c => [1, 2, 2, 1, 2, 1, 2, 1],
        :g => [2, 1, 3, 1, 2, 3, 1, 2]))
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = fill(0.1, built.layout.total)
    native = post_q(u)
    @test native ≈ -22.189852822758517 rtol = 1e-12
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    @test Float64(compiled(Reactant.to_rarray(u))) ≈ native rtol = 1e-9
    q = prepare_sampler(built, bound, u; backend = _RJ_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    @test val ≈ native rtol = 1e-12
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    @test Float64(rval) ≈ native rtol = 1e-9
    @test Array(rgrad) ≈ g rtol = 1e-9
end

# Synthetic grouped linear-PK program (two subjects, a repeated-dose
# schedule): bound plan, sampler-cut kernel and a deterministic point.
function _rj_grouped_pk()
    plan = lower_rkppl(Meta.parse("begin\nsigma ~ Exponential(1.0)\n" *
            "b0 ~ Normal(0.0, 1.0)\nlog_Vc = b0\n" *
            "pk_sched = linear_pk_schedule(obs = (:subj, :time), " *
            "dose = (:dsubj, :dtime, :damt))\n" *
            "@plate conc for s in 1:kernel_nsub_conc\n" *
            " read_locs = linear_pk_read_locs(pk_sched, log_Vc, log_Vc, " *
            "log_Vc, log_Vc, log_Vc)\n" *
            " mu = read_locs[pk_sched.obs_map]\n" *
            " dv .~ Normal.(mu, sigma)\n mu\nend\nend"),
        (:subj, :time, :dsubj, :dtime, :damt, :dv))
    cols = Dict{Symbol,AbstractVector}(
        :subj => [1, 1, 2, 2], :time => [96.0, 120.0, 0.0, 5.0],
        :dsubj => [1, 1, 1, 1, 2], :dtime => [0.0, 24.0, 48.0, 72.0, 0.0],
        :damt => [100.0, 100.0, 100.0, 100.0, 50.0],
        :dv => [10.0, 8.0, 0.5, 7.0])
    bound = bind_data(plan, cols; dims = Dict{Symbol,Int}(:kernel_nsub_conc => 2))
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = Vector{Float64}(0.1 .* randn(Xoshiro(20260917), built.layout.total))
    return (; post_q, u)
end

@testset "Reactant ladder 2: grouped PK rejects the unrolled fallback" begin
    fx = _rj_grouped_pk()
    @test isfinite(fx.post_q(fx.u))
    @test_throws "compiled PK recurrences are disabled" Reactant.@code_hlo fx.post_q(
        Reactant.to_rarray(fx.u))
    # The refusal names the live blocker (scan.md PK adapter note), not the
    # closed issue #13.
    err = try
        Reactant.@code_hlo fx.post_q(Reactant.to_rarray(fx.u))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("scan.md", sprint(showerror, err))
    @test !occursin("issues/13", sprint(showerror, err))
end
