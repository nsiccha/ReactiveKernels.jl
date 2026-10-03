using Test, Distributions, DifferentiationInterface, Enzyme
using ReactiveKernels, ReactiveKernelsPPL

# Synthetic schedules only. The independent oracle uses the mathematical
# basis and Distributions densities, rather than generated prior nodes.
const _EVENT_TEST_ALT = merge(linear_pk_log_f, :(slope ~ Normal(0, 0.5)))
const _EVENT_TEST_REF = merge(linear_pk_log_f, :(reference_dose = 2))

# A default call resolves in the defining module, with positional and earlier
# keyword arguments in scope, just as it does in a Julia function signature.
module EventKeywordDefaults
using ReactiveKernelsPPL
scale_for(x) = 2x
@rkppl draw(x; sd=scale_for(x), center=sd/2) = begin
    s ~ Normal(center, sd)
    return s
end

end
const _EVENT_KEYWORD_DRAW = EventKeywordDefaults.draw

@testset "submodel keyword defaults preserve lexical arguments" begin
    for (call, sd) in ((:(_EVENT_KEYWORD_DRAW(2.0)), 4.0),
                       (:(_EVENT_KEYWORD_DRAW(2.0; sd=6.0)), 6.0))
        model = RKPPLModel(quote
            draw ~ $call
            y .~ Normal.(draw, 1)
        end, @__MODULE__)
        plan = model() | (; y=[0.4])
        built = build_kernel(plan)
        u = [0.3]
        @test Base.invokelatest(prepare_query(built, plan, :prior), u) ≈
            logpdf(Normal(sd/2, sd), u[1])
    end
end

function _event_test_data(G)
    Dict{Symbol,Any}(
        :subj => repeat(collect(1:G); inner=2),
        :time => repeat([1.0, 3.0], G),
        :dsubj => repeat(collect(1:G); inner=3),
        :dtime => repeat([0.0, 0.0, 2.0], G),
        :damt => [j + 0.2s for s in 1:G for j in (1.0, 2.0, 4.0)],
        :dv => [0.3 + 0.1sin(i) for i in 1:2G],
        :age_s => collect(range(-0.5, 0.5; length=G)))
end

function _event_test_ast(; alternate=false, k=5, c=1.5, chain=true)
    file = chain ? "99_plate_50_grouped_pk_logf.jl" : "50_kernel_grouped_pk_logf.jl"
    ast = Meta.parse(read(joinpath(@__DIR__, "corpus", file), String))
    model = RKPPLModel(ast, @__MODULE__)
    name = alternate ? :_EVENT_TEST_ALT : :linear_pk_log_f
    return merge(model, :(log_F ~ $name(pk_sched; k=$k, c=$c)))
end

function _event_test_fixture(G; kwargs...)
    data = _event_test_data(G)
    plan = _event_test_bound(_event_test_ast(; kwargs...), data)
    return plan, build_kernel(plan)
end

_event_test_bound(model, data) =
    model(; (k => v for (k, v) in data if k !== :dv)...) | (; dv = data[:dv])

function _event_test_curve(built, plan, u)
    names = sort!(collect(keys(plan.columns)))
    bound = NamedTuple{Tuple(names)}(Tuple(plan.columns[n] for n in names))
    q = Base.invokelatest(prepare, built.spec;
        have=(:unconstrained, names...), bound, want=:log_F)
    return Base.invokelatest(q, u)
end

function _event_test_oracle(built, plan, u; k=5, c=1.5, slope_sd=1, reference_dose=1)
    nt = constrain(built.layout, u)
    v = nt.log_F
    x = linear_pk_op_log_dose(plan.columns[:pk_sched_op_type],
        plan.columns[:pk_sched_op_amount]; reference_dose)
    PHI, lambda = hsgp_basis(x; k, c)
    floor = maximum(hsgp_rho_floors(lambda))
    curve = v.slope .* x .+ PHI * (hsgp_sqrt_spd(lambda, v.sigma, v.rho) .* v.z)
    prior = logpdf(Normal(0, slope_sd), v.slope) +
        logpdf(truncated(LogNormal(0, 1), floor, Inf), v.rho) +
        logpdf(truncated(Normal(0, 1), 0, Inf), v.sigma) +
        sum(logpdf.(Normal(0, 1), v.z)) + logpdf(Exponential(1), nt.sigma)
    for name in propertynames(nt)
        name in (:sigma, :log_F) && continue
        prior += logpdf(Normal(0, 1), getproperty(nt, name))
    end
    return curve, prior
end

@testset "event LP: explicit library priors and merge" begin
    for alternate in (false, true)
        plan, built = _event_test_fixture(2; alternate)
        @test isempty(plan.event_lps)
        @test isempty(plan.hsgp_bases)
        @test length(plan.parameters) == 14
        @test length(plan.array_parameters) == 1
        @test built.layout.total == 19
        names = coordinate_names(built.layout)
        @test Symbol("log_F.slope") in names
        @test Symbol("log_F.rho") in names
        @test Symbol("log_F.sigma") in names
        @test count(n -> startswith(string(n), "log_F.z."), names) == 5
        u = [0.1cos(i) for i in 1:built.layout.total]
        curve, prior = _event_test_oracle(built, plan, u; slope_sd=alternate ? 0.5 : 1)
        @test _event_test_curve(built, plan, u) ≈ curve rtol=1e-12
        @test Base.invokelatest(prepare_query(built, plan, :prior), u) ≈ prior rtol=1e-12
        # Simultaneous unequal doses must retain separate operations when
        # each dose's bioavailability has its own nonlinear curve value.
        @test count(==(ReactiveKernelsPPL.LINEAR_EVENT_DOSE), plan.columns[:pk_sched_op_type]) == 6
        ex = sprint(show, kernel_expr(plan, built.layout))
        @test occursin("normal", ex) && occursin("lognormal", ex)
        @test !occursin("slope_log_F", ex)
        q = prepare_sampler(built, plan, u; backend=AutoEnzyme(; mode=Enzyme.Reverse))
        value, grad = sampler_value_and_gradient!(q, similar(u), u)
        fd = similar(u)
        h = cbrt(eps(Float64))
        for i in eachindex(u)
            up, down = copy(u), copy(u)
            up[i] += h
            down[i] -= h
            fd[i] = (Base.invokelatest(q.kernel, up) - Base.invokelatest(q.kernel, down)) / (2h)
        end
        @test isfinite(value)
        @test grad ≈ fd rtol=1e-5 atol=1e-7
    end
    # Body rewriting keeps the original library value read-only.
    @test any(st -> st == :(slope ~ Normal(0, 1)), linear_pk_log_f.body.args)
    # Keyword defaults and supplied basis dimensions both bind ordinary
    # expressions; no hidden parameter-generating construct remains.
    p, b = _event_test_fixture(2; k=3, c=2.0)
    @test b.layout.total == 17
    u = fill(0.2, b.layout.total)
    curve, prior = _event_test_oracle(b, p, u; k=3, c=2.0)
    @test _event_test_curve(b, p, u) ≈ curve
    @test Base.invokelatest(prepare_query(b, p, :prior), u) ≈ prior
    # Both schedule-chain and explicit-plate spellings use the same body;
    # omitting both keyword arguments exercises the declared defaults.
    for chain in (false, true)
        model = merge(_event_test_ast(; chain), :(log_F ~ linear_pk_log_f(pk_sched)))
        p = _event_test_bound(model, _event_test_data(2))
        b = build_kernel(p)
        @test b.layout.total == 19
        u = fill(0.2, b.layout.total)
        curve, prior = _event_test_oracle(b, p, u)
        @test _event_test_curve(b, p, u) ≈ curve
        @test Base.invokelatest(prepare_query(b, p, :prior), u) ≈ prior
    end
    model = merge(_event_test_ast(), :(log_F ~ _EVENT_TEST_REF(pk_sched)))
    p = _event_test_bound(model, _event_test_data(2))
    b = build_kernel(p)
    u = fill(0.2, b.layout.total)
    curve, prior = _event_test_oracle(b, p, u; reference_dose=2)
    @test _event_test_curve(b, p, u) ≈ curve
    @test Base.invokelatest(prepare_query(b, p, :prior), u) ≈ prior
    dose = ReactiveKernelsPPL.LINEAR_EVENT_DOSE
    read = ReactiveKernelsPPL.LINEAR_EVENT_READ
    @test linear_pk_op_log_dose([dose, read, dose], [1.0, 0.0, 4.0]) ≈
        [0, log(4)/2, log(4)]
    # Refused: an assignment must not mint unstated parameters (10ldrvz).
    @test_throws SurfaceLoweringError lower_rkppl(quote
        log_F = linear_pk_log_f(pk_sched; k=5)
    end, ())
    legacy = ReactiveKernelsPPL._with(p;
        event_lps=[LinearPKEventLPSpec(:retired, :s, 5, 1.5, nothing, :legacy)])
    # Refused: old hand plans must not bypass explicit priors (10ldrvz).
    @test_throws ContractValidationError validate_structure(legacy)
end

@testset "PK event axis follows position, independent of value name" begin
    data = _event_test_data(2)
    original = _event_test_ast()
    source = read(joinpath(@__DIR__, "corpus", "99_plate_50_grouped_pk_logf.jl"), String)
    renamed = RKPPLModel(Meta.parse(replace(source, r"\blog_F\b" => "bioavailability")),
        @__MODULE__)
    plans = map(m -> _event_test_bound(m, data), (original, renamed))
    built = map(build_kernel, plans)
    names = map(b -> coordinate_names(b.layout), built)
    lookup = Dict(n => i for (i, n) in enumerate(first(names)))
    positions = [lookup[Symbol(replace(string(n), "bioavailability." => "log_F."))]
        for n in last(names)]
    u = [0.1cos(i) for i in eachindex(first(names))]
    qs = map((b, p) -> prepare_query(b, p, :sampler), built, plans)
    @test Base.invokelatest(last(qs), u[positions]) ≈ Base.invokelatest(first(qs), u)
    @test Symbol("bioavailability.slope") in last(names)
    sampler = prepare_sampler(last(built), last(plans), u[positions];
        backend=AutoEnzyme(; mode=Enzyme.Reverse))
    value, grad = sampler_value_and_gradient!(sampler, similar(u), u[positions])
    h = cbrt(eps(Float64))
    fd = map(eachindex(u)) do i
        up, down = copy(u[positions]), copy(u[positions])
        up[i] += h
        down[i] -= h
        (Base.invokelatest(last(qs), up) - Base.invokelatest(last(qs), down)) / (2h)
    end
    @test isfinite(value)
    @test grad ≈ fd rtol=1e-5 atol=1e-7
end
