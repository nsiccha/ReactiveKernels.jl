# Response locations are values (standing principles 3 and 10a,
# decision 09hav95 increment 2). Helpers come from test_generator,
# test_surface, test_bare_location and test_ordinal_explicit.
using LinearAlgebra

_vl_program(body) = Meta.parse("begin; $body; end")
_vl_probe(built) = [0.27 * sin(1.3i) for i in 1:built.layout.total]

@testset "scalar value locations: density and derivatives" begin
    y = [0.3, -1.2, 2.1, 0.7]
    D = Distributions
    sterm = D.logpdf(D.Exponential(1), 1.3) + log(1.3)
    lterm = D.logpdf(D.Normal(), 0.4)
    # The literal, named definition, inline call and bound number follow
    # Julia; a scalar sum with dots broadcasts the computed value too.
    cases = [
        ("y .~ Normal.(0.3, s)", Dict{Symbol,Any}(:y => y),
            (; s = 1.3), 0.3, 0.0),
        ("m = 0.3; y .~ Normal.(m, s)", Dict{Symbol,Any}(:y => y),
            (; s = 1.3), 0.3, 0.0),
        ("y .~ Normal.(c, s)", Dict{Symbol,Any}(:y => y, :c => 0.3),
            (; s = 1.3), 0.3, 0.0),
        ("lm ~ Normal(0,1); m = exp(lm); y .~ Normal.(m, s)",
            Dict{Symbol,Any}(:y => y), (; lm = 0.4, s = 1.3), exp(0.4), lterm),
        ("lm ~ Normal(0,1); y .~ Normal.(exp(lm), s)",
            Dict{Symbol,Any}(:y => y), (; lm = 0.4, s = 1.3), exp(0.4), lterm),
        ("lm ~ Normal(0,1); m = lm .+ 0.3; y .~ Normal.(m, s)",
            Dict{Symbol,Any}(:y => y), (; lm = 0.4, s = 1.3), 0.7, lterm),
        ("lm ~ Normal(0,1); v = lm .+ 0.3; m = v; y .~ Normal.(m, s)",
            Dict{Symbol,Any}(:y => y), (; lm = 0.4, s = 1.3), 0.7, lterm),
        ("lm ~ Normal(0,1); b ~ Normal(0,1); m = lm .+ b; " *
            "y .~ Normal.(m, s)", Dict{Symbol,Any}(:y => y),
            (; lm = 0.4, b = 0.2, s = 1.3), 0.6,
            lterm + D.logpdf(D.Normal(), 0.2)),
        ("lm ~ Normal(0,1); y .~ Normal.(lm + 0.3, s)",
            Dict{Symbol,Any}(:y => y), (; lm = 0.4, s = 1.3), 0.7, lterm),
        ("m ~ Exponential(1); alias = m; y .~ Normal.(alias, s)",
            Dict{Symbol,Any}(:y => y), (; m = 0.4, s = 1.3), 0.4,
            D.logpdf(D.Exponential(1), 0.4) + log(0.4)),
    ]
    for (body, data, q, mu, prior) in cases
        prog = _vl_program("s ~ Exponential(1); $body")
        bound, built, kern, lay = _bare_query(prog, data)
        if haskey(q, :m)
            term = only(only(bound.predictors).terms)
            @test term.kind === InterceptTerm
            @test term.options.parameter === :m
        else
            @test all(t.kind === ComposedTerm for p in bound.predictors for t in p.terms)
        end
        @test Base.invokelatest(kern, unconstrain(lay, _value_q(lay, q))) ≈
            prior + sterm + sum(D.logpdf.(D.Normal(mu, 1.3), y)) rtol = 1e-12
        _check_gradient(built.spec, bound, _vl_probe(built))
    end
    # Naming, pins and the @plate twin compose with scalar data.
    dot = lower_rkppl(_vl_program("y .~ Normal.(c, 1.0)"), (; y, c = 0.3); conditioned = (; y, c = 0.3))
    plate = lower_rkppl(quote
        @plate for i in eachindex(y)
            y[i] ~ Normal(c, 1.0)
        end
    end, (; y, c = 0.3); conditioned = (; y, c = 0.3))
    # The loop keeps its authored indices (explicit-index observations,
    # `85e2f5a1`); apart from that selection the plans agree, and both bind
    # to the same density.
    @test only(plate.responses).range == :(y[eachindex(y)])
    @test plate.indexed_observations == Set([:y])
    @test _plans_equal(dot, ReactiveKernelsPPL._with(plate; responses =
        [ReactiveKernelsPPL._with(r; range = nothing) for r in plate.responses]))
    for twin in (dot, plate)
        b = bind_data(twin, (; y, c = 0.3))
        @test _query(build_kernel(b).spec, b, :likelihood, Float64[]) ≈
            sum(D.logpdf.(D.Normal(0.3, 1), y))
    end
    model = @rkppl begin
        m ~ Normal(0, 1)
        y .~ Normal.(m, 1.0)
    end
    pinned = Base.merge(model, (; m = 0.3))() | (; y)
    built = build_kernel(pinned)
    @test _query(built.spec, pinned, :likelihood, Float64[]) ≈
        sum(D.logpdf.(D.Normal(0.3, 1), y))
    # One definition may locate responses with different observation axes.
    data = Dict{Symbol,Any}(:y => y, :z => y[1:2])
    bound, built, kern, lay = _bare_query(_vl_program("lm ~ Normal(0,1); " *
        "m = exp(lm); y .~ Normal.(m,1.0); z .~ Normal.(m,1.0)"), data)
    @test length(bound.predictors) == 2
    @test Base.invokelatest(kern, unconstrain(lay,
        _value_q(lay, (; lm = 0.4)))) ≈ lterm +
        sum(D.logpdf.(D.Normal(exp(0.4), 1), y)) +
        sum(D.logpdf.(D.Normal(exp(0.4), 1), data[:z]))
    # Signed affine aliases retain their ordinary parameter declaration.
    for rhs in ("a", "-a", "+a", "alias")
        alias = rhs == "alias" ? "alias = -a; " : ""
        p = lower_rkppl(_vl_program("a ~ Normal(0,1); $alias" *
            "m = $rhs; y .~ Normal.(m,1.0)"), (:y,); conditioned = (:y,))
        @test only(only(p.predictors).terms).kind === InterceptTerm
        @test only(p.parameters).name === :a
    end
end

# The same five location shapes in every leveled/joint family.
const _VL_LOCATIONS = ("a", "x", "0.3", "m", "exp(a)")
_vl_means(loc, nt, data) = loc == "x" ? data[:x] :
    fill(loc == "0.3" ? 0.3 : loc == "a" ? nt.a : exp(nt.a), length(data[:y]))

@testset "leveled value locations: likelihood oracles and derivatives" begin
    data = Dict{Symbol,Any}(:y => [1, 2, 3, 2], :x => [0.5, -0.2, 0.1, 0.7])
    D = Distributions
    for loc in _VL_LOCATIONS
        prog = _vl_program("a ~ Normal(0,1); m = exp(a); " *
            "y .~ CategoricalLogit.($loc, 0.2)")
        bound, built, _, _ = _bare_query(prog, data)
        @test [p.name for p in bound.predictors] == [:y_eta_1, :y_eta_2]
        u = _vl_probe(built); nt = constrain(built.layout, u)
        mus = _vl_means(loc, nt, data)
        want = sum(zip(mus, data[:y])) do (mu, y)
            w = [1.0, exp(mu), exp(0.2)]
            D.logpdf(D.Categorical(w ./ sum(w)), y)
        end
        @test _query(built.spec, bound, :likelihood, u) ≈ want rtol = 1e-12
        loc == "m" && _check_gradient(built.spec, bound, u)
        for (head, decl, F, oracle) in (
                ("OrderedLogistic.($loc, Ref(c))", "c ~ Ordered(Normal(0,1),2)",
                    _oe_logistic, _oe_cumulative_lp),
                ("Ordinal.(Cumulative(), ProbitLink(), $loc, Ref(c))",
                    "c ~ Ordered(Normal(0,1),2)", _oe_probit, _oe_cumulative_lp),
                ("Ordinal.(StoppingRatio(), LogitLink(), $loc, Ref(c))",
                    "c[1:2] .~ Normal.(0,1)", _oe_logistic, _oe_stopping_lp))
            prog = _vl_program("a ~ Normal(0,1); m = exp(a); $decl; y .~ $head")
            bound, built, _, _ = _bare_query(prog, data)
            u = _vl_probe(built); nt = constrain(built.layout, u)
            mus = _vl_means(loc, nt, data)
            want = sum(oracle(F, mu, nt.c, y) for (mu, y) in zip(mus, data[:y]))
            @test _query(built.spec, bound, :likelihood, u) ≈ want rtol = 1e-12
            loc == "m" && _check_gradient(built.spec, bound, u)
        end
    end
end

@testset "joint value locations: Distributions oracle and derivatives" begin
    data = Dict{Symbol,Any}(:y => [0.1, 0.3, -0.2, 0.7],
        :z => [0.5, -0.2, 0.1, 0.3], :x => [0.5, -0.2, 0.1, 0.7])
    for loc in _VL_LOCATIONS
        prog = _vl_program("a ~ Normal(0,1); m = exp(a); " *
            "L_scales[1:2] .~ Exponential.(1); L_L_corr ~ LKJCholesky(2,2); L = L_scales .* L_L_corr; " *
            "[y,z] ~ MvNormalCholesky([$loc,0.2],L)")
        bound, built, _, _ = _bare_query(prog, data)
        @test [p.name for p in bound.predictors] == [:y_joint_1, :z_joint_2]
        u = _vl_probe(built); nt = constrain(built.layout, u)
        mus = _vl_means(loc, nt, data)
        L = Diagonal(nt.L_scales) * nt.L_L_corr
        want = sum(Distributions.logpdf(Distributions.MvNormal([mu, 0.2], L * L'),
            [y, z]) for (mu, y, z) in zip(mus, data[:y], data[:z]))
        @test _query(built.spec, bound, :likelihood, u) ≈ want rtol = 1e-12
        loc == "m" && _check_gradient(built.spec, bound, u)
    end
end

# Programs whose response locations are values, with the observed `levels`
# and the `optimize` mode Reactant compiles them with.
function _vl_increment2_programs(levels)
    return [
        ("s ~ Exponential(1); y .~ Normal.(c,s)",
            Dict{Symbol,Any}(:y => [0.1, -0.3, 0.7], :c => 0.3), :only_enzyme),
        ("a ~ Normal(0,1); m = exp(a); y .~ Normal.(m,1.2)",
            Dict{Symbol,Any}(:y => [0.1, -0.3, 0.7]), true),
        ("a ~ Normal(0,1); y .~ Normal.(exp(a),1.2)",
            Dict{Symbol,Any}(:y => [0.1, -0.3, 0.7]), true),
        ("a ~ Normal(0,1); y .~ CategoricalLogit.(a,0.2)",
            Dict{Symbol,Any}(:y => levels), true),
        ("a ~ Normal(0,1); c ~ Ordered(Normal(0,1),2); " *
            "y .~ OrderedLogistic.(a,Ref(c))", Dict{Symbol,Any}(:y => levels), true),
        ("a ~ Normal(0,1); c[1:2] .~ Normal.(0,1); " *
            "y .~ Ordinal.(StoppingRatio(),LogitLink(),a,Ref(c))",
            Dict{Symbol,Any}(:y => levels), true),
        ("a ~ Normal(0,1); L_scales[1:2] .~ Exponential.(1); L_L_corr ~ LKJCholesky(2,2); L = L_scales .* L_L_corr; " *
            "[y,z] ~ MvNormalCholesky([a,0.2],L)",
            Dict{Symbol,Any}(:y => [0.1, -0.3, 0.7], :z => [0.4, 0.7, -0.2]), true),
    ]
end

# The same columns repeated three times; numbers stay as they are.
_vl_bigger(data) = Dict{Symbol,Any}(k => v isa AbstractVector ? repeat(v, 3) : v
    for (k, v) in data)

@testset "value location increment 2: data-length invariance" begin
    # Keep every observed-level branch populated by several irregular rows:
    # singleton lanes and arithmetic index sets take different compiler paths.
    levels = _kinv_levels(3, 9)
    for ys in (levels, repeat(levels, 3))
        @test all(!_kinv_arithmetic(findall(==(k), ys)) for k in 1:3)
    end
    for (body, data, _) in _vl_increment2_programs(levels)
        bigger = _vl_bigger(data)
        prog = _vl_program(body)
        small_plan = bind_data(lower_rkppl(prog, data; conditioned = data), data)
        large_plan = bind_data(lower_rkppl(prog, bigger; conditioned = bigger), bigger)
        @test _kinv_program(small_plan) == _kinv_program(large_plan)
    end
end
