using Distributions: Normal, Exponential, logpdf
using ReactiveKernelsPPL
using Test

# Derived responses (G3): `.~` over a deterministic definition
# (`ly = log.(earn)` then `ly .~ Normal.(mu, s)`) — the logearn/logmesquite
# shape. Cross-file helpers come from test_generator.jl (`_GEN_BACKEND`)
# and test_prior_vocab.jl (`_pv_query`, `_pv_posterior`, `_pv_enzyme_check`),
# both included before this file in runtests.jl.

const _DR_M1 = quote
    b1 ~ Flat()
    b2 ~ Flat()
    s ~ Exponential(1)
    ly = log.(earn)
    mu = b1 .+ b2 .* x
    ly .~ Normal.(mu, s)
end
const _DR_EARN = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
const _DR_X = [0.5, -1.0, 1.5, 0.0, -0.5, 1.0]
_dr_cols() = Dict{Symbol,AbstractVector}(
    :earn => copy(_DR_EARN), :x => copy(_DR_X))

function _dr_m1_oracle(b1::Real, b2::Real, s::Real)
    ly = log.(_DR_EARN)
    mu = b1 .+ b2 .* _DR_X
    ll = sum(logpdf(Normal(m, s), y) for (m, y) in zip(mu, ly))
    # Flat coefficient priors contribute exactly 0.0.
    return ll + logpdf(Exponential(1), s) + log(s)
end

@testset "derived response admission" begin
    plan = lower_rkppl(_DR_M1, (:earn, :x); conditioned = (:earn, :x))
    r = only(plan.responses)
    @test r.response === :ly
    @test r.label === :ly_resp
    @test [d.name for d in plan.derived] == [:ly]
    @test [p.name for p in plan.parameters] == [:b1, :b2, :s]
    @test isempty(plan.population_priors)
    fams = Dict(p.name => p.family for p in plan.parameters)
    @test fams == Dict(:b1 => :flat, :b2 => :flat, :s => :exponential)
    # Forward order lowers identically.
    fwd = lower_rkppl(quote
            b1 ~ Flat()
            b2 ~ Flat()
            s ~ Exponential(1)
            ly .~ Normal.(mu, s)
            mu = b1 .+ b2 .* x
            ly = log.(earn)
        end, (:earn, :x); conditioned = (:earn, :x))
    @test [rr.response for rr in fwd.responses] == [:ly]
    @test [d.name for d in fwd.derived] == [:ly]
    # Bind materializes the response from bound data.
    bound = bind_data(plan, _dr_cols())
    @test bound.columns[:ly] ≈ log.(_DR_EARN)
    @test bound.roles[:ly] === :response
    @test bound.n_obs == 6
end

@testset "derived response fail-closed battery" begin
    cases = (
        ("scalar definition",
            quote
                t = 2.0
                mu = b1 .+ b2 .* x
                t .~ Normal.(mu, s)
            end,
            "t is a scalar definition"),
        ("second observation",
            quote
                ly = log.(earn)
                mu = b1 .+ b2 .* x
                ly .~ Normal.(mu, s)
                ly .~ Normal.(mu, s)
            end,
            "already observed by a `.~` response"),
        ("predictor definition",
            quote
                mu = b1 .+ b2 .* x
                mu .~ Normal.(mu, s)
            end,
            "response mu is a predictor definition"),
        ("levels over definition",
            quote
                c = x .* 2
                mu = c[g] .+ b .* x
                c[levels(g)] .~ Normal.(0, 1)
            end,
            "sizes a coefficient prior but `c` is a deterministic definition"),
        ("direct sampled read",
            quote
                s ~ Exponential(1)
                ly = earn .* s
                mu = b1 .+ b2 .* x
                ly .~ Normal.(mu, s)
            end,
            "response ly is a predictor definition"),
        ("scalar tilde over definition",
            quote
                ly = log.(earn)
                mu = b1 .+ b2 .* x
                ly ~ Normal(0, 1)
            end,
            "defined twice"),
        ("design matrix LHS",
            quote
                X = hcat(ones(length(x)), x)
                mu = X * b
                X .~ Normal.(mu, s)
            end,
            "is a design matrix"),
    )
    for (label, prog, msg) in cases
        if label in ("scalar definition", "design matrix LHS")
            # capability: scalar/matrix data values and valid indexed derived observations (P10a 0dejlw1; todo `1qlbn5b`).
            declarations = label == "design matrix LHS" ? quote
                b[axes(X, 2)] .~ Normal.(0, 1)
                s ~ Exponential(1)
            end : quote
                b1 ~ Normal(0, 1)
                b2 ~ Normal(0, 1)
                s ~ Exponential(1)
            end
            @test_broken (lower_rkppl(Expr(:block, declarations.args..., prog.args...), (:earn, :x, :g); conditioned = (:earn, :x, :g)); true)
        else
            err = try
                lower_rkppl(prog, (:earn, :x, :g); conditioned = (:earn, :x, :g))
                nothing
            catch e
                e
            end
            # refused: duplicate/ambiguous observed LHS or parameter-dependent observed value (single assignment; data-only observation contract, P3/P9).
            @test err isa SurfaceLoweringError
            @test occursin(msg, sprint(showerror, err))
        end
    end
end

@testset "derived response bind" begin
    plan = lower_rkppl(_DR_M1, (:earn, :x); conditioned = (:earn, :x))
    # A caller column the model derives is a shadowing bug, not data.
    err = try
        bind_data(plan, Dict{Symbol,AbstractVector}(:earn => copy(_DR_EARN),
                :x => copy(_DR_X), :ly => copy(_DR_EARN)))
        nothing
    catch e
        e
    end
    # refused: caller data shadows a value the model derives (single assignment, P3).
    @test err isa ContractValidationError
    @test occursin("drop it from bind_data", sprint(showerror, err))
    # A parameter-dependent response definition is rejected during lowering.
    perr = try
        pplan = lower_rkppl(quote
                b1 ~ Flat()
                b2 ~ Flat()
                s ~ Exponential(1)
                t = s * 2
                ly = earn .* t
                mu = b1 .+ b2 .* x
                ly .~ Normal.(mu, s)
            end, (:earn, :x); conditioned = (:earn, :x))
        bind_data(pplan, _dr_cols())
        nothing
    catch e
        e
    end
    # refused: observed data cannot depend on a sampled value (P9, bind-time observation contract).
    @test perr isa SurfaceLoweringError
    @test occursin("response ly is a predictor definition", sprint(showerror, perr))
    # Transitive data-only chains materialize (centering included).
    cplan = lower_rkppl(quote
            b1 ~ Flat()
            b2 ~ Flat()
            s ~ Exponential(1)
            l2 = log.(earn)
            zc = x .- mean(x)
            ly = l2 .+ zc
            mu = b1 .+ b2 .* x
            ly .~ Normal.(mu, s)
        end, (:earn, :x); conditioned = (:earn, :x))
    cbound = bind_data(cplan, _dr_cols())
    using_mean = sum(_DR_X) / length(_DR_X)
    @test cbound.columns[:ly] ≈ log.(_DR_EARN) .+ (_DR_X .- using_mean)
    # Materialized values flow into response validation: a Float-derived
    # Bernoulli response fails exactly like a raw Float column.
    berr = try
        bplan = lower_rkppl(quote
                b1 ~ Flat()
                b2 ~ Flat()
                yb = earn .- 1.0
                mu = b1 .+ b2 .* x
                yb .~ BernoulliLogit.(mu)
            end, (:earn, :x); conditioned = (:earn, :x))
        bind_data(bplan, _dr_cols())
        nothing
    catch e
        e
    end
    # refused: these observed Bernoulli values are outside {0,1} (response domain, P3).
    @test berr isa ContractValidationError
    @test occursin("Bernoulli response must be Bool or 0/1 integers",
        sprint(showerror, berr))
end

@testset "derived response values vs oracles" begin
    _, _, kern, lay = _pv_query(_DR_M1, _dr_cols())
    @test _pv_posterior(kern, lay, (b1 = 0.5, b2 = -0.25, s = 1.3)) ≈
        _dr_m1_oracle(0.5, -0.25, 1.3) rtol = 1e-12
end

@testset "derived response Enzyme gradients" begin
    _pv_enzyme_check(_DR_M1, _dr_cols(), (b1 = 0.5, b2 = -0.25, s = 1.3))
end

# A definition reading only data that a `.~` response observes is that
# observation value, whatever shape lowering can prove: an undotted module
# call's result, an elementwise use of one, or a reduction. `bind_data`
# evaluates the definition once and validates its value as the response,
# exactly like the same value bound as data under that name; a number is one
# observation (snag rkppl-derived-re-e4466bad). A literal-only definition
# (`t = 2.0`) remains the separate capability case in the battery above.
module _DRCells
flatten_cells(cells) = reduce(vcat, cells; init = Float64[])
const calls = Ref(0)
counted_flatten(cells) = (calls[] += 1; reduce(vcat, cells; init = Float64[]))
label_of(cells) = "one label"
end

const _DRC_RAW = [[0.6, 0.2], Float64[], [1.0, 0.8, 0.3]]
const _DRC_FLAT = [0.6, 0.2, 1.0, 0.8, 0.3]
const _DRC_CALL = quote
    a ~ Normal(0, 1)
    s ~ Exponential(1)
    y = flatten_cells(raw)
    y .~ Normal.(a, s)
end
# The identity gather a consumer wrote while the call above was refused.
const _DRC_GATHER = quote
    a ~ Normal(0, 1)
    s ~ Exponential(1)
    values = flatten_cells(raw)
    rows = Base.collect(Base.eachindex(values))
    y = getindex.(Ref(values), rows)
    y .~ Normal.(a, s)
end
_drc_prior(a, s) = logpdf(Normal(0, 1), a) + logpdf(Exponential(1), s) + log(s)
_drc_oracle(a, s, y) = _drc_prior(a, s) + sum(logpdf.(Normal(a, s), y); init = 0.0)

function _drc_kernel(prog, data; conditioned = (:y,))
    plan = lower_rkppl(prog, data; mod = _DRCells, conditioned)
    bound = bind_data(plan, data isa NamedTuple ? data : (; raw = deepcopy(_DRC_RAW)))
    built = build_kernel(bound)
    return plan, bound, built, prepare_query(built, bound, :sampler)
end
_drc_at(kern, built, q) = Base.invokelatest(kern, unconstrain(built.layout, q))

@testset "derived response from a data-only module call" begin
    raw = deepcopy(_DRC_RAW)
    for data in ((; raw), (:raw,))
        plan, bound, built, kern = _drc_kernel(_DRC_CALL, data)
        @test [r.response for r in plan.responses] == [:y]
        @test :y in [d.name for d in plan.derived]
        @test bound.columns[:y] == _DRC_FLAT
        @test bound.roles[:y] === :response
        @test bound.n_obs == 5
        @test coordinate_names(built.layout) == [:a, :s]
        for q in ((a = 0.3, s = 0.7), (a = -0.4, s = 1.6))
            @test _drc_at(kern, built, q) ≈ _drc_oracle(q.a, q.s, _DRC_FLAT) rtol = 1e-12
        end
    end
    @test raw == _DRC_RAW
    # Same density as the identity-gather spelling at the same points.
    _, _, gbuilt, gkern = _drc_kernel(_DRC_GATHER, (; raw = deepcopy(_DRC_RAW)))
    _, _, cbuilt, ckern = _drc_kernel(_DRC_CALL, (; raw = deepcopy(_DRC_RAW)))
    for q in ((a = 0.3, s = 0.7), (a = 1.1, s = 0.4))
        @test _drc_at(ckern, cbuilt, q) ≈ _drc_at(gkern, gbuilt, q) rtol = 1e-14
    end
    # Evaluated once, at bind; the kernel never calls it again.
    _DRCells.calls[] = 0
    _, _, kbuilt, kkern = _drc_kernel(quote
            a ~ Normal(0, 1)
            s ~ Exponential(1)
            y = counted_flatten(raw)
            y .~ Normal.(a, s)
        end, (:raw,))
    @test _DRCells.calls[] == 1
    for q in ((a = 0.3, s = 0.7), (a = -0.2, s = 1.2))
        @test _drc_at(kkern, kbuilt, q) ≈ _drc_oracle(q.a, q.s, _DRC_FLAT) rtol = 1e-12
    end
    @test _DRCells.calls[] == 1
    # Empty cells give zero observations; the priors remain.
    _, ebound, ebuilt, ekern = _drc_kernel(_DRC_CALL,
        (; raw = [Float64[], Float64[]]))
    @test ebound.columns[:y] == Float64[]
    @test _drc_at(ekern, ebuilt, (a = 0.3, s = 0.7)) ≈ _drc_prior(0.3, 0.7) rtol = 1e-12
end

# A range over a derived response selects entries as a range over the same
# value bound as data does, and a `@plate` loop over it writes that range
# (snag rkppl-nested-one-931ad60f).
@testset "derived response: ranges and plate cells" begin
    q = (a = 0.3, s = 0.7)
    for lhs in (:(y[1:5]), :(y[eachindex(y)]), :(y[axes(y, 1)]), :(y[:]))
        _, bound, built, kern = _drc_kernel(quote
                a ~ Normal(0, 1)
                s ~ Exponential(1)
                y = flatten_cells(raw)
                $lhs .~ Normal.(a, s)
            end, (:raw,))
        @test bound.columns[:y] == _DRC_FLAT
        @test _drc_at(kern, built, q) ≈ _drc_oracle(q.a, q.s, _DRC_FLAT) rtol = 1e-12
    end
    _, cbound, cbuilt, ckern = _drc_kernel(quote
            a ~ Normal(0, 1)
            s ~ Exponential(1)
            y = flatten_cells(raw)
            @plate for i in eachindex(y)
                y[i] ~ Normal(a, s)
            end
        end, (:raw,))
    for q in ((a = 0.3, s = 0.7), (a = -0.4, s = 1.6))
        @test _drc_at(ckern, cbuilt, q) ≈ _drc_oracle(q.a, q.s, _DRC_FLAT) rtol = 1e-12
    end
    u = unconstrain(cbuilt.layout, q)
    sampler = prepare_sampler(cbuilt, cbound, u; backend = _GEN_BACKEND)
    g = similar(u)
    sampler_value_and_gradient!(sampler, g, u)
    @test g ≈ _findiff_grad(w -> Base.invokelatest(ckern, w), u) rtol = 1e-5 atol = 1e-7
    # refused: a partial range observes only some entries, exactly as it is
    # for the same values bound as data (whole-response observations,
    # provisional user decision 1uhcm3b).
    partial(lhs) = quote
        a ~ Normal(0, 1)
        s ~ Exponential(1)
        $lhs .~ Normal.(a, s)
    end
    derived = try
        _drc_kernel(Expr(:block, :(y = flatten_cells(raw)), partial(:(y[1:4])).args...), (:raw,))
        nothing
    catch e
        e
    end
    bound = try
        bind_data(lower_rkppl(partial(:(y[1:4])), (:y,); conditioned = (:y,)), (; y = copy(_DRC_FLAT)))
        nothing
    catch e
        e
    end
    @test derived isa ContractValidationError
    @test bound isa ContractValidationError
    @test replace(sprint(showerror, derived), r"\s+" => " ") ==
        replace(sprint(showerror, bound), r"\s+" => " ")
end

@testset "derived response: elementwise and reduction values" begin
    # An elementwise function of a module call's result.
    _, _, lbuilt, lkern = _drc_kernel(quote
            a ~ Normal(0, 1)
            s ~ Exponential(1)
            y = log.(flatten_cells(raw))
            y .~ Normal.(a, s)
        end, (:raw,))
    @test _drc_at(lkern, lbuilt, (a = -0.5, s = 0.9)) ≈
        _drc_oracle(-0.5, 0.9, log.(_DRC_FLAT)) rtol = 1e-12
    # A reduction is one observation, as the same number bound as data.
    _, rbound, rbuilt, rkern = _drc_kernel(quote
            a ~ Normal(0, 1)
            s ~ Exponential(1)
            y = sum(flatten_cells(raw))
            y .~ Normal.(a, s)
        end, (:raw,))
    @test rbound.columns[:y] == [sum(_DRC_FLAT)]
    @test _drc_at(rkern, rbuilt, (a = 0.3, s = 0.7)) ≈
        _drc_oracle(0.3, 0.7, [sum(_DRC_FLAT)]) rtol = 1e-12
end

@testset "derived response from a module call: bind contract" begin
    plan = lower_rkppl(_DRC_CALL, (:raw,); mod = _DRCells, conditioned = (:y,))
    # refused: caller data shadows a value the model derives (single assignment, P3).
    err = try
        bind_data(plan, (; raw = deepcopy(_DRC_RAW), y = copy(_DRC_FLAT)))
        nothing
    catch e
        e
    end
    @test err isa ContractValidationError
    @test occursin("drop it from bind_data", sprint(showerror, err))
    # refused: a String is not an observation value (a response observes a number or an array, P10a).
    lplan = lower_rkppl(quote
            a ~ Normal(0, 1)
            s ~ Exponential(1)
            y = label_of(raw)
            y .~ Normal.(a, s)
        end, (:raw,); mod = _DRCells, conditioned = (:y,))
    lerr = try
        bind_data(lplan, (; raw = deepcopy(_DRC_RAW)))
        nothing
    catch e
        e
    end
    @test lerr isa ContractValidationError
    @test occursin(r"derived response y evaluated to .*String; a response observes a number or an array",
        sprint(showerror, lerr))
end

@testset "derived response from a module call: Enzyme gradients" begin
    plan = lower_rkppl(_DRC_CALL, (:raw,); mod = _DRCells, conditioned = (:y,))
    bound = bind_data(plan, (; raw = deepcopy(_DRC_RAW)))
    built = build_kernel(bound)
    kern = prepare_query(built, bound, :sampler)
    u = unconstrain(built.layout, (a = 0.3, s = 0.7))
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    v, _ = sampler_value_and_gradient!(prep, g, u)
    @test v ≈ _drc_oracle(0.3, 0.7, _DRC_FLAT) rtol = 1e-12
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
end

# A derived response binds the same whichever law observes it: a built-in
# family or a caller-owned sampling law (`LogDensity`, rkppl-use §2). Module
# calls, inline and named index gathers and reductions lower, bind and
# evaluate as their `Normal` twins, exactly like the same values bound as
# data (snag rkppl-derived-re-3096acdf).
module _DRLaws
using ReactiveKernelsPPL: LogDensity
flatten_cells(cells) = reduce(vcat, cells; init = Float64[])
positions(rows) = reduce(vcat, rows; init = Int[])
const calls = Ref(0)
counted_positions(rows) = (calls[] += 1; reduce(vcat, rows; init = Int[]))
normal_lpdf(y, mu, s) = -log(2pi) / 2 - log(s) - ((y - mu) / s)^2 / 2
end

const _DRL_FLAT = [0.6, 0.2, 1.0, 0.8, 0.3]
const _DRL_ROWS = [[3, 1], Int[], [5, 2, 4]]
const _DRL_JOINED = _DRL_FLAT[reduce(vcat, _DRL_ROWS)]
_drl_data() = (; raw = deepcopy(_DRC_RAW), flat = copy(_DRL_FLAT),
    rows = deepcopy(_DRL_ROWS), order = reduce(vcat, _DRL_ROWS))
const _DRL_LAWS = (normal = :(Normal.(a, s)),
    logdensity = :(LogDensity.(normal_lpdf, a, s)))
# Each definition with the value it binds as the response.
const _DRL_DEFS = (
    call = (quote y = flatten_cells(raw) end, _DRC_FLAT),
    inline_index = (quote y = flat[positions(rows)] end, _DRL_JOINED),
    named_index = (quote idx = positions(rows); y = flat[idx] end, _DRL_JOINED),
    data_index = (quote y = flat[order] end, _DRL_JOINED),
    reduction = (quote y = sum(flatten_cells(raw)) end, [sum(_DRC_FLAT)]),
)
_drl_program(def, law) = Expr(:block, :(a ~ Normal(0, 1)), :(s ~ Exponential(1)),
    def.args..., :(y .~ $law))

function _drl_kernel(prog, data = _drl_data(); lowered = data)
    plan = lower_rkppl(prog, lowered; mod = _DRLaws, conditioned = (:y,))
    bound = bind_data(plan, data)
    built = build_kernel(bound)
    return plan, bound, built, prepare_query(built, bound, :sampler)
end

@testset "derived response observed by a caller-owned law" begin
    data = _drl_data()
    for (name, (def, value)) in pairs(_DRL_DEFS), lowered in (data, keys(data))
        at = map(_DRL_LAWS) do law
            _, bound, built, kern = _drl_kernel(_drl_program(def, law); lowered)
            @test bound.columns[:y] == value
            @test coordinate_names(built.layout) == [:a, :s]
            [_drc_at(kern, built, q) for q in ((a = 0.3, s = 0.7), (a = -0.4, s = 1.6))]
        end
        @test at.normal ≈ [_drc_oracle(0.3, 0.7, value), _drc_oracle(-0.4, 1.6, value)] rtol = 1e-12
        # Choosing the law changes neither legality nor the density.
        @test at.logdensity ≈ at.normal rtol = 1e-12
    end
    @test data == _drl_data()
    # Evaluated once, at bind; the kernel never calls it again.
    _DRLaws.calls[] = 0
    _, cbound, cbuilt, ckern = _drl_kernel(_drl_program(
        quote y = flat[counted_positions(rows)] end, _DRL_LAWS.logdensity))
    @test _DRLaws.calls[] == 1
    @test _drc_at(ckern, cbuilt, (a = 0.3, s = 0.7)) ≈ _drc_oracle(0.3, 0.7, _DRL_JOINED) rtol = 1e-12
    @test _DRLaws.calls[] == 1
    # Pointwise densities follow the bound response.
    u = unconstrain(cbuilt.layout, (a = 0.3, s = 0.7))
    pw = Base.invokelatest(prepare_query(cbuilt, cbound, :pointwise), u).y
    @test pw ≈ _DRLaws.normal_lpdf.(_DRL_JOINED, 0.3, 0.7) rtol = 1e-12
    # Rebinding evaluates the definition over the new data.
    rows2 = [[2], [1, 3]]
    rebound = bind_data(cbound, (; data..., rows = rows2))
    @test rebound.columns[:y] == _DRL_FLAT[[2, 1, 3]]
    # refused: caller data shadows a value the model derives (single
    # assignment, P3), with the same message under either law.
    messages = map(_DRL_LAWS) do law
        plan = lower_rkppl(_drl_program(_DRL_DEFS.inline_index[1], law), keys(data);
            mod = _DRLaws, conditioned = (:y,))
        err = try
            bind_data(plan, (; data..., y = copy(_DRL_JOINED)))
            nothing
        catch e
            e
        end
        @test err isa ContractValidationError
        sprint(showerror, err)
    end
    @test occursin("drop it from bind_data", messages.logdensity)
    @test messages.logdensity == messages.normal
    # A number observed through a caller-owned law is one observation, as
    # the same number is through a built-in family.
    for law in _DRL_LAWS
        prog = Expr(:block, :(a ~ Normal(0, 1)), :(s ~ Exponential(1)), :(y .~ $law))
        _, nbound, nbuilt, nkern = _drl_kernel(prog, (; y = 0.6))
        @test nbound.columns[:y] == [0.6]
        @test _drc_at(nkern, nbuilt, (a = 0.3, s = 0.7)) ≈ _drc_oracle(0.3, 0.7, [0.6]) rtol = 1e-12
    end
    # A gathered response holding one array per index observes each index's
    # entries in a dotted `@plate` cell; the gather's inputs are not
    # observation operands.
    for law in _DRL_LAWS
        cells = [[0.6, 0.2], [1.0], Float64[]]
        _, gbound, gbuilt, gkern = _drl_kernel(quote
                a ~ Normal(0, 1)
                s ~ Exponential(1)
                y = cells[perm]
                @plate for i in eachindex(y)
                    y[i] .~ $law
                end
            end, (; cells, perm = [3, 1, 2]))
        @test gbound.columns[:y] == cells[[3, 1, 2]]
        @test _drc_at(gkern, gbuilt, (a = 0.3, s = 0.7)) ≈
            _drc_oracle(0.3, 0.7, [0.6, 0.2, 1.0]) rtol = 1e-12
    end
end

@testset "derived response observed by a caller-owned law: Enzyme gradients" begin
    _, bound, built, kern = _drl_kernel(_drl_program(_DRL_DEFS.inline_index[1],
        _DRL_LAWS.logdensity))
    u = unconstrain(built.layout, (a = 0.3, s = 0.7))
    prep = prepare_sampler(built, bound, u; backend = _GEN_BACKEND)
    g = similar(u)
    v, _ = sampler_value_and_gradient!(prep, g, u)
    @test v ≈ _drc_oracle(0.3, 0.7, _DRL_JOINED) rtol = 1e-12
    @test isapprox(g, _findiff_grad(w -> Base.invokelatest(kern, w), u);
        rtol = 1e-5, atol = 1e-7)
end
