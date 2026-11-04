module GeometryPropertyTests
using Test, ReactiveKernels, ReactiveKernelsPPL
import DifferentiationInterface as DI
import Enzyme

struct PropertyLaw{G,A}
    geometry::G
    location::A
end
struct GetterLaw end
GetterLaw(g, a) = PropertyLaw(g, a)

@kernel coordinates(u, shape, geometry, a) = begin
    return u
end
@kernel jacobian(u, shape, geometry, a) = begin
    return 0.0
end
@kernel density(w, geometry, a) = begin
    residual = w .- a
    return -length(w)*log(2pi)/2 - sum(abs2, residual)/2
end
function geometry(args, shape, getter)
    extent = getter ? :(getproperty($(args[1]), :ncoordinates)) :
        :($(args[1]).ncoordinates)
    dims = isempty(shape) ? (extent,) : shape
    ParameterGeometry(dims, extent; support=:real, constrain=coordinates,
        unconstrain=coordinates, logjac=jacobian)
end
ReactiveKernelsPPL.sampling_geometry(::Type{PropertyLaw}, args, shape) =
    geometry(args, shape, false)
ReactiveKernelsPPL.sampling_geometry(::Type{GetterLaw}, args, shape) =
    geometry(args, shape, true)
ReactiveKernelsPPL.sampling_distribution_graph(::Type{PropertyLaw}, args::Tuple) = density
ReactiveKernelsPPL.sampling_distribution_graph(::Type{GetterLaw}, args::Tuple) = density

# Deliberately use a property with no field of the same name: the evaluator
# must call getproperty, not getfield, including a caller's newer method.
struct Dimensions
    count::Int
end
Base.getproperty(x::Dimensions, s::Symbol) =
    s === :ncoordinates ? getfield(x, :count) : getfield(x, s)
make_geometry(n) = (; ncoordinates=n)
make_geometries(ns) = make_geometry.(ns)
nested_geometry(n) = (; inner=make_geometry(n))

function fixture(ns; cells=false, getter=false, record=false, nested=false,
        property_shape=false, argument_property=false)
    head = getter ? :GetterLaw : :PropertyLaw
    raw = ns
    y = cells ? [[0.1i for i in 1:n] for n in ns] : [0.1i for i in 1:ns]
    data = (; raw, y)
    definition = cells ? :(geometry = make_geometries(raw)) :
        record ? :(geometry = Dimensions(raw)) :
        nested ? :(geometry = nested_geometry(raw).inner) :
        argument_property ? :(holder = nested_geometry(raw)) :
        :(geometry = make_geometry(raw))
    dimension = property_shape ? :(geometry.ncoordinates) : :(length(y))
    statements = cells ? quote
        @plate for i in eachindex(geometry)
            w[i] ~ $head(geometry[i], a)
            y[i] .~ Normal.(w[i], 1)
        end
    end : argument_property ? quote
        w[1:$dimension] ~ $head(holder.inner, a)
        y .~ Normal.(w, 1)
    end : quote
        w[1:$dimension] ~ $head(geometry, a)
        y .~ Normal.(w, 1)
    end
    ast = Expr(:block, :(a ~ Normal(0, 1)), definition, statements.args...)
    bound = bind_data(lower_rkppl(ast, data; mod=@__MODULE__, conditioned=(:y,)), data)
    built = build_kernel(bound)
    w = cells ? [[0.04i for i in 1:n] for n in ns] : [0.04i for i in 1:ns]
    u = unconstrain(built.layout, (; a=0.2, w))
    (; data, bound, built, u, cells)
end

function oracle(fx, u)
    values = constrain(fx.built.layout, u)
    w = fx.cells ? reduce(vcat, values.w; init=Float64[]) : values.w
    y = fx.cells ? reduce(vcat, fx.data.y; init=Float64[]) : fx.data.y
    prior = -(length(w)+1)*log(2pi)/2 - values.a^2/2 - sum(abs2, w .- values.a)/2
    likelihood = -length(y)*log(2pi)/2 - sum(abs2, y .- w)/2
    (; prior, likelihood, sampler=prior+likelihood)
end
nodes(ex) = ex isa Expr ? 1 + sum(nodes, ex.args; init=0) : 0
holds_module(x) = x isa Module || (x isa Expr && any(holds_module, x.args)) ||
    (x isa QuoteNode && holds_module(x.value))
rebuilds_binding(x) = x isa Expr && ((x.head === :call && x.args[1] == GlobalRef(Core, :GlobalRef)) ||
    any(rebuilds_binding, x.args))

function check(fx; reverse=false)
    saved = deepcopy((fx.data, fx.u))
    source = kernel_expr(fx.bound, fx.built.layout)
    # A program holds no module, so a deep copy of anything holding it works
    # (snags `prepared-query-h-b7bd890a`, `rkppl-layout-pla-fb46f0ee`); the
    # printed source names a law argument's binding and replays.
    @test !holds_module(source)
    replay = ReactiveKernelsPPL._eval_kernel_def(
        Meta.parse(string(Base.remove_linenums!(deepcopy(source)))))
    replay_query = prepare_query((; spec=replay, layout=fx.built.layout), fx.bound, :sampler)
    queries = map(cut -> prepare_query(fx.built, fx.bound, cut), (:prior, :likelihood, :sampler))
    # Per-chain copies deep-copy prepared queries.
    copies = map(deepcopy, queries)
    @test model_view(fx.built; bound=fx.bound).source isa Expr
    catalogue = density_factors(fx.built, fx.bound)
    priors = select_factors(catalogue; kind=:prior, site=:w)
    @test length(priors) == (fx.cells ? length(fx.data.y) : 1)
    @test factor_coordinates(priors) == collect(2:length(fx.u))
    if length(fx.u) > 1
        # A joint factor keeps its live parent and the selected cell's block;
        # property names and geometry data never become sampled dependencies.
        selected = fx.cells ? select_factors(catalogue; kind=:prior, site=:w,
            entries=findfirst(!isempty, fx.data.y)) : priors
        @test factor_support(selected; u0=fx.u) == [1; factor_coordinates(selected)]
    end
    for shift in (0.0, 0.07)
        u = fx.u .+ shift
        expected = oracle(fx, u)
        for (q, value) in zip(queries, values(expected))
            @test Base.invokelatest(q, u) ≈ value
        end
        for (q, value) in zip(copies, values(expected))
            @test Base.invokelatest(q, u) ≈ value
        end
        @test Base.invokelatest(replay_query, u) ≈ expected.sampler
        @test unconstrain(fx.built.layout, constrain(fx.built.layout, u)) ≈ u
        @test logjac(fx.built.layout, u) == 0.0
    end
    if reverse
        @test Base.get_extension(ReactiveKernels, :ReactiveKernelsEnzymeExt) !== nothing
        sampler = prepare_sampler(fx.built, fx.bound, fx.u;
            backend=DI.AutoEnzyme(; mode=Enzyme.Reverse))
        value, gradient = sampler_value_and_gradient!(sampler, similar(fx.u), fx.u)
        @test value ≈ oracle(fx, fx.u).sampler
        expected = map(eachindex(fx.u)) do i
            plus, minus = copy(fx.u), copy(fx.u)
            plus[i] += 1e-5
            minus[i] -= 1e-5
            (oracle(fx, plus).sampler-oracle(fx, minus).sampler)/2e-5
        end
        @test gradient ≈ expected atol=2e-8 rtol=2e-7
    end
    @test (fx.data, fx.u) == saved
    return (nodes(source), nodes(code_expr(queries[3])))
end

@testset "bound properties in parameter geometry" begin
    # Control: explicit getproperty already supports the same Julia operation.
    control = fixture(3; getter=true)
    check(control)
    counts = map((0, 3, 19)) do n
        fx = fixture(n)
        @test fx.built.layout.total == n+1
        check(fx; reverse=true)
    end
    @test counts[2] == counts[3]
    check(fixture(3; record=true); reverse=true)
    check(fixture(3; nested=true))
    check(fixture(3; property_shape=true))
    # A resolved binding in the law's own argument (`holder.inner` calls
    # `Base.getproperty`): the geometry recipe's source rebuilds the binding.
    argument = fixture(3; argument_property=true)
    @test any(r -> rebuilds_binding(r.source), argument.built.spec.graph.recipes)
    check(argument; reverse=true)
    for ns in ([2, 1, 3], [0, 2], Int[])
        fx = fixture(ns; cells=true)
        @test fx.built.layout.total == sum(ns)+1
        check(fx; reverse=true)
    end
    small = check(fixture([2, 1]; cells=true))
    large = check(fixture(fill(3, 19); cells=true))
    @test small == large
end

# A law whose arguments hold resolved bindings: a property read of an indexed
# record, a matrix built from array literals, and a value the density reads.
struct ChannelLaw{G,F,A}
    geometry::G
    factor::F
    location::A
end
@kernel channel_coordinates(u, shape, geometry, factor, a) = begin
    return u
end
@kernel channel_jacobian(u, shape, geometry, factor, a) = begin
    return 0.0
end
@kernel channel_density(w, geometry, factor, a) = begin
    residual = w .- a * factor[2, 2]
    return -length(w)*log(2pi)/2 - sum(abs2, residual)/2
end
ReactiveKernelsPPL.sampling_geometry(::Type{ChannelLaw}, args, shape) = begin
    extent = :($(args[1]).ncoordinates)
    ParameterGeometry(isempty(shape) ? (extent,) : shape, extent; support=:real,
        constrain=channel_coordinates, unconstrain=channel_coordinates,
        logjac=channel_jacobian)
end
ReactiveKernelsPPL.sampling_distribution_graph(::Type{ChannelLaw}, args::Tuple) = channel_density
make_records(ns) = [(; geometry=make_geometry(n)) for n in ns]

function channel_fixture(ns; cells)
    y = cells ? [[0.1i for i in 1:n] for n in ns] : [0.1i for i in 1:ns]
    data = (; raw=ns, y)
    statements = cells ? quote
        records = make_records(raw)
        @plate for i in eachindex(records)
            w[i] ~ ChannelLaw(records[i].geometry, hcat([1.0, 0.4], [0.0, 0.9]), a)
            y[i] .~ Normal.(w[i], 1)
        end
    end : quote
        records = make_records([raw])
        w[1:length(y)] ~ ChannelLaw(records[1].geometry, hcat([1.0, 0.4], [0.0, 0.9]), a)
        y .~ Normal.(w, 1)
    end
    ast = Expr(:block, :(a ~ Normal(0, 1)), statements.args...)
    bound = bind_data(lower_rkppl(ast, data; mod=@__MODULE__, conditioned=(:y,)), data)
    built = build_kernel(bound)
    w = cells ? [[0.04i for i in 1:n] for n in ns] : [0.04i for i in 1:ns]
    (; data, bound, built, u=unconstrain(built.layout, (; a=0.2, w)), cells)
end

function channel_oracle(fx, u)
    values = constrain(fx.built.layout, u)
    w = fx.cells ? reduce(vcat, values.w; init=Float64[]) : values.w
    y = fx.cells ? reduce(vcat, fx.data.y; init=Float64[]) : fx.data.y
    (-(length(w)+1)*log(2pi)/2 - values.a^2/2 - sum(abs2, w .- 0.9values.a)/2) +
        (-length(y)*log(2pi)/2 - sum(abs2, y .- w)/2)
end

@testset "every object holding a law's resolved-binding arguments deep-copies" begin
    # The objects a per-chain or per-plan copy reaches: the layout (and the
    # cell expressions in its `bound_exprs`), the plan (and its derived
    # statements), the built program and every query over them
    # (snag `rkppl-layout-pla-fb46f0ee`).
    for fx in (channel_fixture([2, 1, 3]; cells=true), channel_fixture(3; cells=false))
        (; bound, built, u) = fx
        source = kernel_expr(bound, built.layout)
        @test rebuilds_binding(source)
        @test !holds_module(source)
        @test !holds_module(built.program)
        @test !any(holds_module, values(built.layout.bound_exprs))
        @test !any(d -> holds_module(getfield(d, :expr)),
            filter(d -> hasfield(typeof(d), :expr), bound.derived))
        # The copies are independent and the program they build evaluates.
        plan_copy, built_copy = deepcopy(bound), deepcopy(built)
        @test built_copy.program !== built.program
        @test built_copy.program == built.program
        @test built_copy.layout.bound_exprs !== built.layout.bound_exprs
        @test built_copy.layout.total == built.layout.total
        expected = channel_oracle(fx, u)
        @test Base.invokelatest(prepare_query(built, bound, :sampler), u) ≈ expected
        @test Base.invokelatest(prepare_query(built_copy, plan_copy, :sampler), u) ≈ expected
        @test Base.invokelatest(prepare_query(build_kernel(plan_copy), plan_copy, :sampler), u) ≈
            expected
        @test Base.invokelatest(deepcopy(prepare_query(built, bound, :sampler)), u) ≈ expected
        # The density factors and their selection keep working through a copy.
        catalogue = density_factors(built, bound)
        @test factor_coordinates(select_factors(deepcopy(catalogue); kind=:prior, site=:w)) ==
            factor_coordinates(select_factors(catalogue; kind=:prior, site=:w))
        # The printed source replays with the same value, through both literal
        # readers: the geometry's constructor and the distribution's endpoint.
        replay = ReactiveKernelsPPL._eval_kernel_def(
            Meta.parse(string(Base.remove_linenums!(deepcopy(source)))))
        @test Base.invokelatest(prepare_query((; spec=replay, layout=built.layout), bound,
            :sampler), u) ≈ expected
    end
    fx = channel_fixture([2, 1, 3]; cells=true)
    (; bound, built, u) = fx
    query = prepare_cell_query(built, bound, (:y,))
    factors = prepare_cell_factors(built, bound, :y; locality=:error)
    query_copy, factors_copy = deepcopy(query), deepcopy(factors)
    @test query_copy.layout.bound_exprs !== query.layout.bound_exprs
    values_at = constrain(built.layout, u)
    for (q, f) in ((query, factors), (query_copy, factors_copy))
        term, outer = prepare_cell_factor_query(f, 1), prepare_cell_factor_query(f, :outer)
        for c in eachindex(fx.data.y)
            y, w = fx.data.y[c], values_at.w[c]
            expected = -length(y)*log(2pi)/2 - sum(abs2, y .- w)/2
            @test Base.invokelatest(q, u, c) ≈ expected
            terms = [term(u, c, t) for t in eachindex(y)]
            @test terms ≈ -log(2pi)/2 .- abs2.(y .- w)./2
            @test outer(u, [sum(terms)], c) ≈ expected
        end
    end
end

@testset "models and submodels deep-copy and share their module" begin
    model = @rkppl begin
        a ~ Normal(0, 1)
        y ~ Normal(a, 1)
    end
    @rkppl shifted(x) = begin
        b ~ Normal(x, 1)
        return b
    end
    model_copy, submodel_copy = deepcopy(model), deepcopy(shifted)
    @test model_copy.mod === model.mod === @__MODULE__
    @test model_copy.ast !== model.ast
    @test model_copy.ast == model.ast
    @test submodel_copy.mod === shifted.mod === @__MODULE__
    @test submodel_copy.body !== shifted.body
    @test submodel_copy.body == shifted.body
    binding = model(; y=0.1)
    binding_copy = deepcopy(binding)
    @test binding_copy.model.mod === binding.model.mod
    @test binding_copy.model.ast == binding.model.ast
    @test binding_copy.data !== binding.data
    # The copy lowers to the same program, with the same value.
    plan = model() | (; y=0.1)
    plan_copy = deepcopy(model)() | (; y=0.1)
    original, copied = build_kernel(plan), build_kernel(plan_copy)
    u = unconstrain(original.layout, (; a=0.3))
    @test Base.invokelatest(prepare_query(copied, plan_copy, :sampler), u) ==
        Base.invokelatest(prepare_query(original, plan, :sampler), u)
end

@testset "geometry property reads keep bound-data validation" begin
    data = (; n=2, y=[0.1, 0.2])
    plan = lower_rkppl(quote
        a ~ Normal(0, 1)
        geometry = make_geometry(n)
        w[1:length(y)] ~ PropertyLaw(geometry, a)
        y .~ Normal.(w, 1)
    end, data; mod=@__MODULE__, conditioned=(:y,))
    for n in (-1, 1.5, true)
        # Existing coordinate-pack invariant: extents are nonnegative integers.
        invalid = (; n, y=data.y)
        err = try
            build_kernel(bind_data(plan, invalid))
        catch e
            e
        end
        @test err isa ContractValidationError
        @test occursin("must be a nonnegative integer", sprint(showerror, err))
    end
    live = quote
        a ~ Normal(0, 1)
        geometry = make_geometry(a)
        w[1:length(y)] ~ PropertyLaw(geometry, a)
        y .~ Normal.(w, 1)
    end
    # A live parameter cannot determine a structural packed dimension.
    err = try
        build_kernel(bind_data(lower_rkppl(live, (; data.y);
            mod=@__MODULE__, conditioned=(:y,)), (; data.y)))
    catch e
        e
    end
    @test err isa ContractValidationError
    @test occursin("must depend only on bound data", sprint(showerror, err))
end
end
