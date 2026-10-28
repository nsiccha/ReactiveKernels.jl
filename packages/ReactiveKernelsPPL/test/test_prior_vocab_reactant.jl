# Compiled (Reactant) checks for the cases in test_prior_vocab.jl.
using Reactant

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _pv_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_pv_reactant_measure, built, bound, post_q, u)
end

function _pv_reactant_measure(built, bound, post_q, u)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    val, _ = sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), native, primal, val, g,
        rval = Float64(rval), rgrad = Array(rgrad))
end

@testset "sampled Uniform bounds native and compiled parity" begin
    sizes = Int[]
    for n in (3, 7)
        fx = _pv_reactant(_PV_SAMPLED_UNIFORM,
            Dict{Symbol,AbstractVector}(:y => fill(0.2, n)))
        @test fx.primal ≈ fx.native rtol = 1e-9
        @test fx.val ≈ fx.native rtol = 1e-12
        @test fx.rval ≈ fx.native rtol = 1e-9
        @test fx.rgrad ≈ fx.g rtol = 1e-8
        push!(sizes, fx.lines)
    end
    @test sizes[1] == sizes[2]
end

@testset "prior vocab under Reactant" begin
    @testset "mixed population" begin
        fx = _pv_reactant(_PV_M1, _pv_cols())
        @test fx.primal ≈ fx.native rtol = 1e-9
        @test fx.val ≈ fx.native rtol = 1e-12
        @test fx.rval ≈ fx.native rtol = 1e-9
        @test fx.rgrad ≈ fx.g rtol = 1e-8
    end
    @testset "uniform plus half" begin
        fx = _pv_reactant(_PV_M3, _pv_cols())
        @test fx.primal ≈ fx.native rtol = 1e-9
        @test fx.val ≈ fx.native rtol = 1e-12
        @test fx.rval ≈ fx.native rtol = 1e-9
        @test fx.rgrad ≈ fx.g rtol = 1e-8
    end
    @testset "traced program is O(1) in n_obs and n_levels" begin
        small = _pv_reactant(_PV_M4, _pv_gcols())
        bigcols = Dict{Symbol,AbstractVector}(
            :y => vcat(_PV_Y, _PV_Y), :x => vcat(_PV_X, _PV_X),
            :g => vcat(_PV_G, _PV_G))
        @test _pv_reactant(_PV_M4, bigcols).lines == small.lines
        morelevels = Dict{Symbol,AbstractVector}(
            :y => vcat(_PV_Y, _PV_Y), :x => vcat(_PV_X, _PV_X),
            :g => vcat(_PV_G, _PV_G .+ 3))
        @test _pv_reactant(_PV_M4, morelevels).lines == small.lines
    end
end

@testset "normalized halves under Reactant" begin
    fx = _pv_reactant(_PV_M5,_pv_cols())
    @test fx.primal ≈ fx.native rtol=1e-9
    @test fx.rgrad ≈ fx.g rtol=1e-8
end

@testset "centered factor priors under Reactant" begin
    # Vector-mu M8: default pipeline (narrowed §7n scope — scalar-mu
    # only).
    fx = _pv_reactant(_PV_M8, _pv_gcols())
    @test fx.primal ≈ fx.native rtol = 1e-9
    @test fx.val ≈ fx.native rtol = 1e-12
    @test fx.rval ≈ fx.native rtol = 1e-9
    @test fx.rgrad ≈ fx.g rtol = 1e-8
    # The centered hyper plate stays one plate: doubling n_obs /
    # n_levels adds no HLO lines (core constraint 1 — robust G1).
    bigcols = Dict{Symbol,AbstractVector}(
        :y => vcat(_PV_Y, _PV_Y), :x => vcat(_PV_X, _PV_X),
        :g => vcat(_PV_G, _PV_G))
    @test _pv_reactant(_PV_M8, bigcols).lines == fx.lines
    morelevels = Dict{Symbol,AbstractVector}(
        :y => vcat(_PV_Y, _PV_Y), :x => vcat(_PV_X, _PV_X),
        :g => vcat(_PV_G, _PV_G .+ 3))
    @test _pv_reactant(_PV_M8, morelevels).lines == fx.lines
end

@testset "mixed flat scalar offset under Reactant" begin
    fx = _pv_reactant(_PV_M9, _pv_cols())
    @test fx.primal ≈ fx.native rtol = 1e-9
    @test fx.val ≈ fx.native rtol = 1e-12
    @test fx.rval ≈ fx.native rtol = 1e-9
    @test fx.rgrad ≈ fx.g rtol = 1e-8
end

@testset "uniform coefficients under Reactant" begin
    fx = _pv_reactant(_PV_M10, _pv_m10_cols())
    @test fx.primal ≈ fx.native rtol = 1e-9
    @test fx.val ≈ fx.native rtol = 1e-12
    @test fx.rval ≈ fx.native rtol = 1e-9
    @test fx.rgrad ≈ fx.g rtol = 1e-8
    # Split-block reassembly is structural (runs, not lanes): doubling
    # n_obs adds no HLO lines (core constraint 1 — robust G1).
    bigcols = Dict{Symbol,AbstractVector}(
        :y => vcat(_PV_YB, _PV_YB), :x => vcat(_PV_X, _PV_X),
        :z => vcat(_PV_Z, _PV_Z))
    @test _pv_reactant(_PV_M10, bigcols).lines == fx.lines
end
