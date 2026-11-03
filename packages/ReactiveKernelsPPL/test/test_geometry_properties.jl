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

function check(fx; reverse=false)
    saved = deepcopy((fx.data, fx.u))
    source = kernel_expr(fx.bound, fx.built.layout)
    # `copy`: a law argument's rebuilt binding holds its module, which
    # `deepcopy` refuses; the printed source names it and replays.
    replay = ReactiveKernelsPPL._eval_kernel_def(
        Meta.parse(string(Base.remove_linenums!(copy(source)))))
    replay_query = prepare_query((; spec=replay, layout=fx.built.layout), fx.bound, :sampler)
    queries = map(cut -> prepare_query(fx.built, fx.bound, cut), (:prior, :likelihood, :sampler))
    # Per-chain copies deep-copy prepared queries, including the authored
    # source a rebuilt argument's module is part of (snag
    # `prepared-query-h-b7bd890a`).
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
    # `Base.getproperty`): the geometry recipe's source holds `Base`.
    argument = fixture(3; argument_property=true)
    @test any(r -> holds_module(r.source), argument.built.spec.graph.recipes)
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
