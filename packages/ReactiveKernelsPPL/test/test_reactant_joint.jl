# Reactant track (brm:tgi:reactant, 2026-09-21): full-program
# `Reactant.@compile` of built RKPPL programs — primal lp parity and
# compiled value+gradient (Enzyme-through-Reactant) parity against the
# native kernels. Ladder: tiny model → joint PK+QT+TGI fixture at
# tiling K=1 (→ K=3/10 in the tiled testsets below).
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

isdefined(@__MODULE__, :_parity_columns) ||
    include(joinpath(@__DIR__, "parity", "joint_parity_fixture.jl"))
isdefined(@__MODULE__, :tiled_columns) ||
    include(joinpath(@__DIR__, "parity", "joint_tiling.jl"))

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
    # (c) compiled value + gradient over the sampler's prepared AD
    q = prepare_sampler(built, bound, u; backend = _RJ_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    @test val ≈ native rtol = 1e-12
    @test g ≈ _rj_findiff(q, u) rtol = 1e-6
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    @test Float64(rval) ≈ native rtol = 1e-9
    @test Array(rgrad) ≈ g rtol = 1e-9
end

# Joint fixture at tiling K: bound plan + built program + sampler-cut kernel
# + a deterministic unconstrained point (the benchmark's point, seed
# 20260917).  `tiled_columns(1)` is the fixture itself (pinned below).
function _rj_joint(K)
    bound = bind_data(final_plan("continuous"), tiled_columns(K);
        dims = Dict(:kernel_nsub_pk_loc => 3K))
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = Vector{Float64}(0.1 .* randn(Xoshiro(20260917), built.layout.total))
    return (; bound, built, post_q, u)
end

@testset "Reactant ladder 2: joint K=1 primal @compile parity" begin
    fx = _rj_joint(1)
    @test fx.built.layout.total == 101
    native = fx.post_q(fx.u)
    # The tiling helper at K=1 reproduces the parity fixture's columns.
    bound_f = bind_data(final_plan("continuous"), _parity_columns("continuous");
        dims = Dict(:kernel_nsub_pk_loc => 3))
    @test prepare_query(build_kernel(bound_f), bound_f, :sampler)(fx.u) == native
    # Full-program compile of the sampler-cut kernel.  The only lowering gap
    # this hit was the PK cell's per-op `log_F[j]` read on the traced
    # bioavailability slice (now `_traced_op_read`, pkcells.jl); everything
    # else — packed reads, gathers, the TGI scan, plates — traces as authored.
    compiled = Reactant.@compile fx.post_q(Reactant.to_rarray(fx.u))
    @test Float64(compiled(Reactant.to_rarray(fx.u))) ≈ native rtol = 1e-9
    u2 = Vector{Float64}(0.1 .* randn(Xoshiro(7), length(fx.u)))
    @test Float64(compiled(Reactant.to_rarray(u2))) ≈ fx.post_q(u2) rtol = 1e-9
end
