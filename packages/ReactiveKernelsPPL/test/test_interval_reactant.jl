# Compiled (Reactant) checks for the cases in test_interval.jl.
using Reactant

# Reactant/XLA value+grad parity at an unconstrained probe (no oracle —
# native vs compiled), plus the traced program size.
function _int_reactant(prog::Expr, cols::Dict{Symbol,AbstractVector})
    plan = lower_rkppl(prog, keys(cols); conditioned = keys(cols))
    bound = bind_data(plan, cols)
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_int_reactant_measure, built, bound, post_q, u)
end

function _int_reactant_measure(built, bound, post_q, u)
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

@testset "interval under Reactant" begin
    prog = quote
        a ~ Normal(0, 1)
        b ~ Normal(0, 1)
        mu = a .+ b .* x
        y .~ interval_censored.(Normal.(mu, s), hi)
        s ~ Exponential(1)
    end
    @testset "gaussian column upper" begin
        fx = _int_reactant(prog, _int_gcols())
        @test fx.primal ≈ fx.native rtol = 1e-9
        @test fx.val ≈ fx.native rtol = 1e-12
        @test fx.rval ≈ fx.native rtol = 1e-9
        @test fx.rgrad ≈ fx.g rtol = 1e-8
    end
    @testset "traced program is O(1) in n_obs" begin
        small = _int_reactant(prog, _int_gcols())
        bigcols = Dict{Symbol,AbstractVector}(:y => vcat(_INT_YG, _INT_YG),
            :x => vcat(_INT_X, _INT_X), :hi => vcat(_INT_HI, _INT_HI))
        large = _int_reactant(prog, bigcols)
        @test small.lines == large.lines
    end
end

# Attempt the Poisson-interval HLO trace, catching the expected upstream
# failure (returns the exception, or `:traced` if Reactant ever closes
# the gap).
function _int_poisson_hlo_attempt()
    plan = lower_rkppl(quote
            a ~ Normal(0, 1)
            b ~ Normal(0, 1)
            eta = a .+ b .* x
            y .~ interval_censored.(Poisson.(exp.(eta)), ub)
        end, (:y, :x, :ub); conditioned = (:y, :x, :ub))
    bound = bind_data(plan, _int_pcols())
    built = build_kernel(bound)
    post_q = prepare_query(built, bound, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_int_try_hlo, post_q, u)
end

function _int_try_hlo(post_q, u)
    try
        Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u))
        return :traced
    catch err
        return err
    end
end

@testset "poisson interval XLA gap is pinned upstream" begin
    # `poisson.cdf` lowers through `SpecialFunctions.gamma_inc`, which
    # has no Reactant tracing rule (the hurdle truncated-Poisson note).
    # If this test starts failing, the gap closed: promote Poisson
    # interval to the trio testset above and delete this pin.
    err = _int_poisson_hlo_attempt()
    # capability: Poisson interval probabilities under XLA (generic AD;
    # todo `0ze68k8`). A tracing failure is a gap, not a forbidden model.
    @test_broken err === :traced
    @test err isa MethodError && err.f === ReactiveKernelsPPL.SpecialFunctions.gamma_inc &&
        any(arg -> arg isa Reactant.TracedRNumber, err.args)
end
