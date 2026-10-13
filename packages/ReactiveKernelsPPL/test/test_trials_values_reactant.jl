# Compiled (Reactant) checks for test_trials_values.jl: a number bound as
# trials compiles to the same value and reverse gradient as native
# execution and as the column spelling, and the traced program does not grow
# with the observation count. (`_TV_BINOMIAL`, `_TV_ZIB`, `_TV_X` and
# `_TV_Y` come from test_trials_values.jl, included first.)
using Enzyme
using Reactant

function _tv_reactant(ast, data)
    values = Dict{Symbol,Any}(pairs(data))
    plan = bind_data(lower_rkppl(ast, values; conditioned = (:y,)), values)
    built = build_kernel(plan)
    post_q = prepare_query(built, plan, :sampler)
    u = [0.3 * sin(1.7i) for i in 1:built.layout.total]
    return Base.invokelatest(_tv_reactant_measure, built, plan, post_q, u)
end

function _tv_reactant_measure(built, plan, post_q, u)
    hlo = repr(Reactant.@code_hlo optimize = false post_q(Reactant.to_rarray(u)))
    native = post_q(u)
    compiled = Reactant.@compile post_q(Reactant.to_rarray(u))
    primal = Float64(compiled(Reactant.to_rarray(u)))
    q = prepare_sampler(built, plan, u; backend = AutoEnzyme(; mode = Enzyme.Reverse))
    g = similar(u)
    sampler_value_and_gradient!(q, g, u)
    cad = compile_ad_value_and_gradient(q.ad, Reactant.to_rarray(u))
    rval, rgrad = cad(Reactant.to_rarray(u))
    return (; lines = count(==('\n'), hlo), native, primal, g,
        rval = Float64(rval), rgrad = Array(rgrad))
end

@testset "a number as trials under Reactant" begin
    for (name, ast) in (("Binomial logit", _TV_BINOMIAL),
            ("ZeroInflatedBinomial", _TV_ZIB))
        @testset "$name" begin
            number = _tv_reactant(ast, (; x = _TV_X, y = _TV_Y, n = 5))
            column = _tv_reactant(ast, (; x = _TV_X, y = _TV_Y, n = fill(5, 8)))
            @test number.primal ≈ number.native rtol = 1e-9
            @test number.rval ≈ number.native rtol = 1e-9
            @test number.rgrad ≈ number.g rtol = 1e-8
            @test number.native ≈ column.native rtol = 1e-13
            @test number.g ≈ column.g rtol = 1e-12
            # The traced program keeps one retained body as rows grow.
            twice = _tv_reactant(ast, (; x = repeat(_TV_X, 2),
                y = repeat(_TV_Y, 2), n = 5))
            @test twice.lines == number.lines
        end
    end
end
