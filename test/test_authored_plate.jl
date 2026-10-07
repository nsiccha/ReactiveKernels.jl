using ReactiveKernels
using LogExpFunctions: log1pexp
using Test

isdefined(@__MODULE__, :AuthoredPlateChains) ||
    include("fixtures/authored_plate_chains.jl")

@testset "Demand-driven authored plate chains" begin
    C = AuthoredPlateChains
    q = [0.7]
    x = collect(range(-1.0, 1.0; length = 4096))
    y = fill(0.3, length(x))
    p = plan(C.chain)
    recipes = copy(p.recipes)
    kernel = prepare(p)
    reference = prepare(C.flat)
    @test kernel(q, x, y) == reference(q, x, y)
    @test kernel.plan === p
    @test p.recipes == recipes
    @test count(r -> r.op isa ReactiveKernels._AuthoredPlateOp, p.recipes) == 2
    @test !occursin("similar", string(code_expr(kernel)))
    shown_cell = ReactiveKernels._readable_recipe_call(
        last(kernel.lowered_recipes), Any[0.2, 0.7])
    @test Core.eval(@__MODULE__, shown_cell) == -0.5 * (0.2 - 0.7)^2
    C.allocated(kernel, q, x, y)
    @test C.allocated(kernel, q, x, y) == 0
    @test kernel(q, Float64[], Float64[]) == 0.0
    @test_throws DimensionMismatch kernel(q, x, y[1:2])

    mu = x .* only(q)
    pw = -0.5 .* (mu .- y).^2
    @test prepare(C.chain; want = :mu)(q, x, y) == mu
    @test prepare(C.chain; want = :pointwise)(q, x, y) == pw
    both = prepare(C.chain; want = (:mu, :pointwise, :total))
    @test both(q, x, y) == (mu, pw, kernel(q, x, y))
    shared = prepare(C.chain; want = (:total, :extra))
    @test first(shared(q, x, y)) == kernel(q, x, y)
    @test last(shared(q, x, y)) ≈ sum(mu) atol = 1e-10
    @test occursin("similar", string(code_expr(shared)))
    cut = prepare(C.chain; have = (:mu, :y), want = :total)
    @test cut(mu, y) == kernel(q, x, y)
    bound = prepare(C.chain; bound = (; x, y))
    @test bound(q) == kernel(q, x, y)
    @test !occursin("similar", string(code_expr(bound)))

    multi = prepare(C.multidimensional)
    a = reshape([1.0, 2.0, 3.0], 3, 1)
    b = reshape([0.5, 1.0], 1, 2)
    observations = fill(0.2, 3, 2)
    scales = reshape([2.0, 3.0], 1, 2)
    @test multi(a, b, observations, scales) ≈
        sum(((a .+ b) .- observations) .* log.(scales))
    # A later vector must not make a scalar-only producer a valid plate.
    @test_throws ArgumentError multi(1.0, 2.0, y, 2.0)
    @test_throws DimensionMismatch multi(ones(2), ones(3), y, 2.0)
    @test multi(zeros(0, 1), ones(1, 2), zeros(0, 2), 2.0) == 0.0
    repeated = prepare(C.repeated)
    @test repeated(x, 2.0) ≈ sum(4 .* x) atol = 1e-10
    @test !occursin("similar", string(code_expr(repeated)))
    atomic = prepare(C.atomic)
    @test atomic(x, y) ≈ sum(y .+ sum(abs2, x))
    @test occursin("similar", string(code_expr(atomic)))
    unused = prepare(C.unused_axis)
    @test unused(x, y) ≈ sum(1 .+ y)
    @test_throws ArgumentError unused(1.0, y)
    @test_throws DimensionMismatch unused(ones(2), y)
end

@testset "authored plate chain: bound data with only Ref-atomic array HAVE" begin
    C = AuthoredPlateChains
    q = [0.7, -0.3]
    x = collect(range(-1.0, 1.0; length = 8))
    y = fill(0.3, length(x))
    middle = 2 .* x .+ q[1]
    pointwise = y .+ middle .^ 2 .+ q[2] .* middle
    ref = sum(pointwise)

    unbound = prepare(C.ref_atomic_chain)
    @test unbound(q, x, y) ≈ ref

    # Binding the raw-data axis arrays must NOT remove the runtime backend
    # marker: `q` is still an active array HAVE even though it is captured
    # atomically, so `prepare(...; bound = (; x, y))` must succeed (it used to
    # throw "an embedded plate requires an array-valued HAVE port").
    bound = prepare(C.ref_atomic_chain; bound = (; x, y))
    @test bound(q) ≈ ref
    @test bound(q) == unbound(q, x, y)
    @test bound.f isa ReactiveKernels._ArrayFunctionPair

    # A demanded intermediate keeps its materialization boundary and stays
    # bound/unbound consistent.
    bound_mid = prepare(C.ref_atomic_chain; want = :middle, bound = (; x, y))
    @test bound_mid(q) == middle
end

@testset "embedded prepared plate: bound axis with untyped live array HAVE" begin
    C = AuthoredPlateChains
    schedule = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]   # nobs = 6 plate-axis rows
    dose = [0.5, 1.5, 2.5]                        # ndoses = 3 ≠ nobs
    plan_axis = collect(1.0:length(schedule))
    ref = 2 .* plan_axis .+ dose[1]

    # Untyped (`Any`) live dose port with a bound schedule-derived axis. This
    # used to throw "an embedded plate requires an array-valued HAVE port in
    # the outer kernel" because the fallback marker search only admitted
    # array-TYPED ports.
    outer = C.embedded_dose_outer_graph()
    unbound = prepare(outer.graph; have = (outer.schedule, outer.dose),
                      want = outer.result)
    @test unbound(schedule, dose) ≈ ref
    bound = prepare(outer.graph; have = (outer.schedule, outer.dose),
                    want = outer.result, bound = [outer.schedule => schedule])
    @test bound.f isa ReactiveKernels._DynamicEmbeddedFunctionPair
    got = bound(dose)
    @test got ≈ ref
    # The plate axis is the bound nobs plan, never the ndoses live dose extent.
    @test length(got) == length(schedule)
    # The live dose dependence is real: changed doses change the result.
    @test bound(dose .+ 1.0) ≈ 2 .* plan_axis .+ (dose[1] + 1.0)
    @test !(bound(dose .+ 1.0) ≈ got)

    # A typed dose port keeps the static backend marker.
    touter = C.embedded_dose_outer_graph(; dose_type = Vector{Float64})
    tbound = prepare(touter.graph; have = (touter.schedule, touter.dose),
                     want = touter.result,
                     bound = [touter.schedule => schedule])
    @test tbound.f isa ReactiveKernels._EmbeddedFunctionPair
    @test tbound(dose) ≈ ref

    # The axis is already bound. A scalar live port needs no runtime array
    # marker; it selects the native body while retaining the bound axis.
    souter = C.embedded_dose_outer_graph(; dose_type = Float64)
    sbound = prepare(
        souter.graph; have = (souter.schedule, souter.dose),
        want = souter.result, bound = [souter.schedule => schedule])
    @test sbound(0.5) ≈ 2 .* plan_axis .+ 0.5
end

@kernel authored_standard_normal() = begin
    logpdf(z::Float64)::Float64 = -0.5 * log(2π) - 0.5 * z^2
end

@kernel authored_standard_cauchy() = begin
    logpdf(z::Float64)::Float64 = -log(π) - log1p(z^2)
end

@kernel authored_location_scale(standard, location::Float64, scale::Float64) = begin
    log_scale::Float64 = log(scale)
    scale::Float64 = exp(log_scale)
    standardized(x::Float64)::Float64 = (x - location) / scale

    logpdf(x::Float64)::Float64 = begin
        z::Float64 = standardized(x)
        standard.logpdf(z) - log_scale
    end
end


@kernel authored_normal = authored_location_scale(authored_standard_normal)
@kernel authored_cauchy = authored_location_scale(authored_standard_cauchy)

@kernel authored_vector_location(location::Vector{Float64}) = begin
    logpdf(x::Vector{Float64})::Float64 =
        -0.5 * sum(abs2, x .- location)
end

const _AUTHORED_PLATE_SCALE_CALLS = Ref(0)
_authored_plate_counted_log(scale) =
    (_AUTHORED_PLATE_SCALE_CALLS[] += 1; log(scale))

# This helper is a transparent RK graph. Its leaf operation is an ordinary
# opaque Julia callable, which carries the normal RK pure-recipe contract.
@kernel authored_counted_scale(scale) = begin
    log_scale::Float64 = _authored_plate_counted_log(scale)
    return log_scale
end

@kernel authored_axis_loglik(x, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        log_scale::Float64 = authored_counted_scale(si)
        residual::Float64 = xi - li
        logpdf::Float64 =
            -0.5 * log(2π) - log_scale - 0.5 * (residual / si)^2
        return logpdf
    end
    return sum(pointwise)
end

@kernel authored_normal_loglik(x::Vector{Float64}, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        authored_normal(li, si).logpdf(xi)
    end
    return sum(pointwise)
end

@kernel authored_normal_both_loglik(x, location, scale, log_scale) = begin
    pointwise = plate(x, location, scale, log_scale) do xi, li, si, log_si
        authored_normal(;
            location = li, scale = si, log_scale = log_si).logpdf(xi)
    end
    return sum(pointwise)
end

@kernel authored_eight_schools_prior(
        θ::Vector{Float64}, μ::Float64, τ::Float64, log_τ::Float64) = begin
    μ_prior::Float64 = authored_normal(0.0, 5.0).logpdf(μ)
    τ_cauchy::Float64 = authored_cauchy(;
        location = 0.0, scale = τ, log_scale = log_τ).logpdf(τ)
    effects_pointwise = plate(θ, μ, τ, log_τ) do θj, μj, τj, log_τj
        authored_normal(;
            location = μj, scale = τj, log_scale = log_τj).logpdf(θj)
    end
    prior::Float64 = μ_prior + τ_cauchy + sum(effects_pointwise)
    return prior
end

@kernel authored_namedtuple_eight_schools_prior(parameters) = begin
    μ::Float64 = parameters.μ
    τ::Float64 = parameters.τ
    θ::AbstractVector{Float64} = parameters.θ
    log_τ::Float64 = log(τ)
    μ_prior::Float64 = authored_normal(0.0, 5.0).logpdf(μ)
    τ_cauchy::Float64 = authored_cauchy(0.0, 5.0).logpdf(τ)
    τ_prior::Float64 = log(2.0) + τ_cauchy
    effects_pointwise = plate(θ, μ, τ, log_τ) do θj, μj, τj, log_τj
        authored_normal(;
            location = μj, scale = τj, log_scale = log_τj).logpdf(θj)
    end
    prior::Float64 = μ_prior + τ_prior + sum(effects_pointwise)
    return prior
end

@kernel authored_namedtuple_observed_loglik(
        parameters, observations, observation_scales) = begin
    μ::Float64 = parameters.μ
    pointwise = plate(μ, observations, observation_scales) do μj, xj, sj
        authored_normal(μj, sj).logpdf(xj)
    end
    return sum(pointwise)
end

@kernel untyped_authored_normal_loglik(x, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        authored_normal(li, si).logpdf(xi)
    end
    return sum(pointwise)
end

# The endpoint METHOD argument accepts a computed expression, exactly as the
# constructor arguments (mean/scale) already do. `.logpdf(log(wi))` lowers to
# the same graph as precomputing `log_w` in a transformed-data node and passing
# the bare port, so a log-response likelihood need not be split off the plate.
@kernel authored_transformed_response_loglik(
        w::Vector{Float64}, location, scale) = begin
    pointwise = plate(w, location, scale) do wi, li, si
        authored_normal(li, si).logpdf(log(wi))
    end
    return sum(pointwise)
end

@kernel authored_cauchy_loglik(x::Vector{Float64}, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        authored_cauchy(li, si).logpdf(xi)
    end
    return sum(pointwise)
end

@kernel authored_vector_loglik(
        x::Vector{Vector{Float64}}, location::Vector{Float64}) = begin
    pointwise = plate(x, Ref(location)) do xi, li
        authored_vector_location(li).logpdf(xi)
    end
    return sum(pointwise)
end


@kernel untyped_authored_vector_loglik(x, location) = begin
    pointwise = plate(x, Ref(location)) do xi, li
        authored_vector_location(li).logpdf(xi)
    end
    return sum(pointwise)
end

@kernel authored_eachcol_sum(matrix, offsets) = begin
    pointwise = plate(eachcol(matrix), offsets) do column, offset
        value::Float64 = sum(column) + offset
        return value
    end
    return sum(pointwise)
end

@kernel authored_eachrow_derived_sum(matrix, offsets) = begin
    pointwise = plate(eachrow(matrix), offsets .+ 1.0) do row, offset
        value::Float64 = sum(row) + offset
        return value
    end
    return sum(pointwise)
end

@kernel authored_ref_derived_sum(x, offsets) = begin
    pointwise = plate(x, Ref(offsets .+ 1.0)) do value, atomic_offsets
        result::Float64 = value + sum(atomic_offsets)
        return result
    end
    return sum(pointwise)
end

_authored_logsumexp(values) = log(sum(exp, values))

@kernel authored_categorical_logit(logits::AbstractVector{Float64}) = begin
    logpdf(observed::Int)::Float64 =
        logits[observed] - _authored_logsumexp(logits)
end

@kernel authored_eachcol_categorical(logits, observed) = begin
    pointwise = plate(eachcol(logits), observed) do column, value
        authored_categorical_logit(column).logpdf(value)
    end
    return sum(pointwise)
end

@kernel authored_vcat_normal(W, b) = begin
    pointwise = plate(vcat(vec(W), b)) do coefficient
        authored_normal(0.0, 1.0).logpdf(coefficient)
    end
    return sum(pointwise)
end

@kernel authored_implicit_typed_plate(x::Vector{Float64}) = begin
    terms = plate(x) do value
        squared::Float64 = value * value
        squared
    end
    total::Float64 = sum(terms)
end

@kernel authored_implicit_untyped_plate(x::Vector{Float64}) = begin
    terms = plate(x) do value
        squared = value * value
        squared
    end
    total::Float64 = sum(terms)
end

@kernel authored_implicit_assignment_plate(x::Vector{Float64}) = begin
    terms = plate(x) do value
        squared::Float64 = value * value
    end
    total::Float64 = sum(terms)
end

# Identity/passthrough cell: the body is the bare loop variable, so the plate's
# distinguished result names an input rather than a recipe output.
@kernel authored_identity_plate(x::Vector{Float64}) = begin
    terms = plate(x) do value
        value
    end
    total::Float64 = sum(terms)
end

# The reporter's exact shape (snag identity-plate-o): an identity plate over
# another plate's output, exposing a passthrough/generated-quantity node.
@kernel authored_identity_over_plate(x::Vector{Float64}) = begin
    mu = plate(x) do value
        doubled::Float64 = 2.0 * value
        doubled
    end
    expected = plate(mu) do m
        m
    end
    total::Float64 = sum(expected)
end

_authored_plate_normal(x, location, scale) =
    -0.5 * log(2π) - log(scale) - 0.5 * ((x - location) / scale)^2
_authored_plate_cauchy(x, location, scale) =
    -log(π) - log(scale) - log1p(((x - location) / scale)^2)

function _authored_plate_allocated(kernel, a, b, c)
    kernel(a, b, c)
    @allocated kernel(a, b, c)
end

_authored_plate_steady_allocated(kernel, a, b, c) =
    @allocated kernel(a, b, c)

struct _AuthoredPlateBackendArray <: AbstractVector{Float64}
    values::Vector{Float64}
end
Base.size(marker::_AuthoredPlateBackendArray) = size(marker.values)
Base.getindex(marker::_AuthoredPlateBackendArray, index::Int) =
    marker.values[index]
ReactiveKernels._requires_tensorized_marker(
    ::_AuthoredPlateBackendArray) = true
ReactiveKernels._batched_call(
        pair::ReactiveKernels._ArrayFunctionPair, ops, args,
        ::_AuthoredPlateBackendArray) = pair.tensorized(ops, args...)

function _authored_plate_head_count(node, head)
    node isa Expr || return 0
    (node.head === head ? 1 : 0) +
        sum(_authored_plate_head_count(child, head) for child in node.args)
end

@testset "authored plate block: implicit multi-statement result" begin
    values = [1.0, 2.0, 3.0]
    expected = values .^ 2

    for spec in (authored_implicit_typed_plate,
                 authored_implicit_untyped_plate,
                 authored_implicit_assignment_plate)
        scalar_plan = plate_body(first(plan(spec).recipes))
        @test [value.name for value in scalar_plan.have] == [:value]
        @test [value.name for value in scalar_plan.want] == [:squared]
        @test [only(recipe.outputs).name for recipe in scalar_plan.recipes] ==
              [:squared]

        total = prepare(spec; have = (:x,), want = :total)
        pointwise = prepare(extract(spec; have = (:x,), want = :terms))
        @test total(values) == sum(expected)
        @test pointwise(values) == expected
        # A materialized cell must produce a concrete container, never a boxed
        # `Vector{Any}` — an untyped cell (metadata result type `Any`) is
        # narrowed to its element type so the node stays promotable at the
        # Reactant host-operand boundary (snag untyped-plate-ce).
        @test pointwise(values) isa Vector{Float64}
    end
end

@testset "authored plate block: identity/passthrough cell" begin
    values = [1.0, 2.0, 3.0]

    # A cell whose body is the bare loop variable names an input as its
    # distinguished result. That result id equals a HAVE id and is absent from
    # the recipe-output `locals`, which previously threw
    # `KeyError` in `_lower_authored_plate_native!` (snag identity-plate-o).
    scalar_plan = plate_body(first(plan(authored_identity_plate).recipes))
    @test [value.name for value in scalar_plan.have] == [:value]
    @test isempty(scalar_plan.recipes)
    # The root cause: the distinguished result canonicalizes to the same value as
    # the sole input, so it never appears among the recipe-output locals.
    @test ReactiveKernels.canon_id(scalar_plan.graph, only(scalar_plan.want).id) ==
          ReactiveKernels.canon_id(scalar_plan.graph, only(scalar_plan.have).id)

    total = prepare(authored_identity_plate; have = (:x,), want = :total)
    pointwise = prepare(
        extract(authored_identity_plate; have = (:x,), want = :terms))
    @test pointwise(values) == values
    @test total(values) == sum(values)

    # The reporter's exact shape: an identity plate over another plate's output.
    doubled = 2.0 .* values
    over_pointwise = prepare(
        extract(authored_identity_over_plate; have = (:x,), want = :expected))
    over_total = prepare(
        authored_identity_over_plate; have = (:x,), want = :total)
    @test over_pointwise(values) == doubled
    @test over_total(values) == sum(doubled)
end

@testset "authored plate block: transparent distribution log-likelihood" begin
    xs = [-1.2, -0.1, 0.7, 1.8]
    location = 0.3
    scale = 1.2

    total_plan = plan(authored_normal_loglik)
    pointwise_spec = extract(authored_normal_loglik; want = :pointwise)
    both_spec = extract(authored_normal_loglik;
                        want = (:pointwise, :__return__))
    pointwise_plan = plan(pointwise_spec)
    both_plan = plan(both_spec)

    @test keys(authored_normal_loglik) ==
        (:x, :location, :scale, :pointwise, :__return__)
    @test only(outputs(authored_normal_loglik)).name === :__return__
    @test length(total_plan.recipes) == 2
    @test length(pointwise_plan.recipes) == 1
    @test length(both_plan.recipes) == 2
    @test count(recipe -> recipe.source isa Expr && recipe.source.head === :do,
                authored_normal_loglik.graph.recipes) == 1
    @test count(recipe -> recipe.source == :(sum(pointwise)),
                authored_normal_loglik.graph.recipes) == 1

    plate_recipe = first(total_plan.recipes)
    scalar_plan = plate_body(plate_recipe)
    scalar_names = Symbol[output.name for recipe in scalar_plan.recipes
                          for output in recipe.outputs]
    @test :log_scale in scalar_names
    @test :standardized in scalar_names
    @test Symbol("standard.logpdf") in scalar_names
    @test :logpdf in scalar_names
    @test occursin("standard.logpdf", dot_source(total_plan))

    total = prepare(authored_normal_loglik)
    pointwise = prepare(pointwise_spec)
    both = prepare(both_spec)
    reference = [_authored_plate_normal(x, location, scale) for x in xs]

    untyped_total = prepare(untyped_authored_normal_loglik)
    @test untyped_total.f isa ReactiveKernels._DynamicEmbeddedFunctionPair
    @test typeof(untyped_total.f).parameters[1] == (1, 2, 3)
    @test untyped_total(xs, location, scale) ≈ sum(reference)
    @test _authored_plate_allocated(
        untyped_total, xs, location, scale) == 0
    @test !occursin("for ", string(untyped_total.f.tensorized_ast))

    parameters = (; μ = location, τ = scale, θ = xs)
    constrained_prior = prepare(
        authored_namedtuple_eight_schools_prior;
        have = :parameters, want = :prior)
    constrained_reference =
        _authored_plate_normal(location, 0.0, 5.0) +
        log(2.0) + _authored_plate_cauchy(scale, 0.0, 5.0) +
        sum(_authored_plate_normal(value, location, scale) for value in xs)
    @test constrained_prior.f isa
          ReactiveKernels._DynamicEmbeddedFunctionPair
    @test typeof(constrained_prior.f).parameters[1] == (1,)
    @test constrained_prior(parameters) ≈ constrained_reference

    scalar_parameters = (; μ = location)
    observed_scales = fill(scale, length(xs))
    observed_total = prepare(authored_namedtuple_observed_loglik)
    observed_pointwise = prepare(extract(
        authored_namedtuple_observed_loglik; want = :pointwise))
    @test typeof(observed_total.f).parameters[1] == (1, 2, 3)
    @test observed_total(scalar_parameters, xs, observed_scales) ≈ sum(
        _authored_plate_normal(xs[i], location, observed_scales[i])
        for i in eachindex(xs))
    observed_pointwise_text = string(code_expr(observed_pointwise))
    @test findfirst("Base.broadcastable", observed_pointwise_text) <
          findfirst("similar", observed_pointwise_text)
    @test_throws DimensionMismatch observed_pointwise(
        scalar_parameters, xs, observed_scales[1:3])

    # Positional KernelSpec application requests the distinguished return.
    @test authored_normal_loglik(xs, location, scale) ≈ sum(reference)
    @test total(xs, location, scale) ≈ sum(reference)
    @test pointwise(xs, location, scale) ≈ reference
    @test both(xs, location, scale) == (pointwise(xs, location, scale),
                                       total(xs, location, scale))

    raw_ast = lower(total_plan)
    raw_ops = Tuple(recipe.op for recipe in total_plan.recipes)
    @test ReactiveKernels.compile(raw_ast)(raw_ops, xs, location, scale) ≈
        sum(reference)

    total_ast = code_expr(total)
    pointwise_ast = code_expr(pointwise)
    both_ast = code_expr(both)
    @test _authored_plate_head_count(total_ast, :for) == 1
    @test _authored_plate_head_count(pointwise_ast, :for) == 1
    @test _authored_plate_head_count(both_ast, :for) == 1
    @test !occursin("similar", string(total_ast))
    @test occursin("similar", string(pointwise_ast))
    @test occursin("similar", string(both_ast))
    @test !occursin("for ", string(total.f.tensorized_ast))
    @test occursin("_tensorized_plate_call", string(total.f.tensorized_ast))
    native_text = string(total_ast)
    @test findfirst("Base.broadcastable", native_text) <
          findfirst("__ops__[", native_text)
    for materializing_ast in (pointwise_ast, both_ast)
        materializing_text = string(materializing_ast)
        @test findfirst("Base.broadcastable", materializing_text) <
              findfirst("similar", materializing_text)
        @test findfirst("Base.broadcastable", materializing_text) <
              findfirst("__ops__[", materializing_text)
    end

    readable = string(ReactiveKernels._readable_expr(total_ast, total))
    @test occursin("standard.logpdf", readable)
    @test !occursin(r"__ops__\[\d+\]", readable)

    locations = [0.1, 0.2, 0.4, 0.5]
    scales = [0.8, 1.0, 1.3, 1.5]
    @test untyped_total(0.25, locations, scale) ≈ sum(
        _authored_plate_normal(0.25, item, scale) for item in locations)

    # A native array in an earlier candidate position must not mask a later
    # backend-traced array (or traced scalar) that requires tensorized lowering.
    native_call = (ops, args...) -> :native
    tensorized_call = (ops, args...) -> :tensorized
    pair = ReactiveKernels._DynamicEmbeddedFunctionPair{
        (1, 2),typeof(native_call),typeof(tensorized_call),Expr}(
            native_call, tensorized_call, Expr(:block))
    single_candidate_pair = ReactiveKernels._DynamicEmbeddedFunctionPair{
        (1,),typeof(native_call),typeof(tensorized_call),Expr}(
            native_call, tensorized_call, Expr(:block))
    typed_pair = ReactiveKernels._EmbeddedFunctionPair{
        1,typeof(native_call),typeof(tensorized_call),Expr}(
            native_call, tensorized_call, Expr(:block))
    backend_locations = _AuthoredPlateBackendArray(locations)
    @test pair((), xs, backend_locations) === :tensorized
    @test pair((), xs, locations) === :native
    @test pair((), (; μ = location, τ = scale), locations) === :native
    @test pair((), Dict(:μ => location, :τ => scale), locations) === :native
    @test pair((), (; μ = location, τ = scale), backend_locations) ===
          :tensorized
    @test pair((), (; θ = xs), locations) === :native
    @test pair((), (; θ = backend_locations), locations) === :tensorized
    @test single_candidate_pair((), (; μ = location, τ = scale)) === :native
    @test single_candidate_pair(
        (), (; μ = location, τ = scale, θ = backend_locations)) ===
          :tensorized
    @test typed_pair((), xs, backend_locations) === :tensorized
    @test typed_pair((), xs, locations) === :native
    zipped_reference = [
        _authored_plate_normal(xs[i], locations[i], scales[i])
        for i in eachindex(xs)
    ]
    @test total(xs, locations, scales) ≈ sum(zipped_reference)
    @test pointwise(xs, locations, scales) ≈ zipped_reference
    @test _authored_plate_allocated(total, xs, locations, scales) == 0
    @test total(xs, locations, scale) ≈
        sum(_authored_plate_normal(xs[i], locations[i], scale)
            for i in eachindex(xs))

    both_have = prepare(authored_normal_both_loglik)
    @test both_have(xs, location, scale, log(scale)) ≈ sum(reference)
    both_have_scalar = plate_body(first(plan(authored_normal_both_loglik).recipes))
    @test [only(recipe.outputs).name for recipe in both_have_scalar.recipes] ==
          [:standardized, Symbol("standard.logpdf"), :logpdf]
    @test !occursin("KernelObjectSpec", string(code_expr(both_have)))

    θ = [0.25 * index for index in 1:8]
    μ = 1.5
    τ = 2.0
    log_τ = log(τ)
    eight_schools_prior = prepare(authored_eight_schools_prior)
    expected_prior =
        _authored_plate_normal(μ, 0.0, 5.0) +
        _authored_plate_cauchy(τ, 0.0, τ) +
        sum(_authored_plate_normal(value, μ, τ) for value in θ)
    @test eight_schools_prior(θ, μ, τ, log_τ) ≈ expected_prior
    eight_schools_plan = plan(authored_eight_schools_prior)
    @test any(recipe -> recipe.source == 0.0, eight_schools_plan.recipes)
    @test any(recipe -> recipe.source == 5.0, eight_schools_plan.recipes)
    @test all(!(recipe.op isa PreparedKernel) for recipe in eight_schools_plan.recipes)
    @test !occursin("KernelObjectSpec", string(code_expr(eight_schools_prior)))

    # Julia broadcast semantics include singleton expansion.
    @test pointwise(xs, locations[1:1], scale) ≈
        [_authored_plate_normal(x, only(locations[1:1]), scale) for x in xs]

    location_grid = reshape(locations[1:3], 1, :)
    scale_grid = reshape(scales[1:2], 1, 1, :)
    grid_reference = _authored_plate_normal.(xs, location_grid, scale_grid)
    @test pointwise(xs, location_grid, scale_grid) ≈ grid_reference
    @test total(xs, location_grid, scale_grid) ≈ sum(grid_reference)

    @test_throws DimensionMismatch total(xs, locations[1:3], scales)
    @test_throws DimensionMismatch total(xs, [locations; 0.6], scales)
end

@testset "authored plate block: computed endpoint method argument" begin
    w = [0.5, 1.5, 2.0, 3.0]
    location = 0.1
    scale = 1.3

    # A computed method argument `log(wi)` produces the SAME graph, and the same
    # numbers, as precomputing the transformed response in a transformed-data
    # node (`workaround_loglik`) and passing the bare port.
    transformed = prepare(authored_transformed_response_loglik)
    precomputed = prepare(untyped_authored_normal_loglik)
    reference = sum(_authored_plate_normal(log(wi), location, scale) for wi in w)

    @test transformed(w, location, scale) ≈ reference
    @test transformed(w, location, scale) ≈ precomputed(log.(w), location, scale)
    @test _authored_plate_allocated(
        transformed, w, location, scale) == 0

    # A bare name that is not a declared caller port is still rejected — the
    # method argument path only gained expression materialization, not the
    # ability to reference an undeclared local.
    @test_throws ArgumentError @macroexpand @kernel _authored_undeclared_arg(
            w::Vector{Float64}, location, scale) = begin
        pointwise = plate(w, location, scale) do wi, li, si
            authored_normal(li, si).logpdf(missing_port)
        end
        return sum(pointwise)
    end
end

@testset "authored plate block: graph-derived broadcast scheduling" begin
    observations = collect(range(-1.0, 1.0; length = 7))
    locations = reshape([-0.4, 0.2, 0.7], 1, :)
    scales = reshape([0.8, 1.1, 1.7, 2.2], 1, 1, :)
    reference = _authored_plate_normal.(observations, locations, scales)

    total = prepare(authored_axis_loglik)
    pointwise = prepare(extract(authored_axis_loglik; want = :pointwise))
    both = prepare(extract(
        authored_axis_loglik; want = (:pointwise, :__return__)))

    scalar_plan = plate_body(first(plan(authored_axis_loglik).recipes))
    scalar_names = [only(recipe.outputs).name for recipe in scalar_plan.recipes]
    @test :log_scale in scalar_names
    @test all(!(recipe.op isa PreparedKernel) for recipe in scalar_plan.recipes)

    # Julia's instantiated broadcast dimensions are the logical axes. The
    # transparent dependency graph proves that `log(scale)` depends only on
    # the third one, so it runs once per scale coordinate, not once per cell.
    for (kernel, expected) in (
            (total, sum(reference)),
            (pointwise, reference))
        _AUTHORED_PLATE_SCALE_CALLS[] = 0
        @test kernel(observations, locations, scales) ≈ expected
        @test _AUTHORED_PLATE_SCALE_CALLS[] == length(scales)
    end
    _AUTHORED_PLATE_SCALE_CALLS[] = 0
    both_result = both(observations, locations, scales)
    @test first(both_result) ≈ reference
    @test last(both_result) ≈ sum(reference)
    @test _AUTHORED_PLATE_SCALE_CALLS[] == length(scales)

    generated = string(code_expr(total))
    @test occursin("_plate_dependency_changed", generated)
    @test _authored_plate_head_count(code_expr(total), :for) == 1
    @test !occursin("similar", generated)
    @test !occursin("for ", string(total.f.tensorized_ast))

    total(observations, locations, scales)
    _AUTHORED_PLATE_SCALE_CALLS[] = 0
    @test _authored_plate_steady_allocated(
        total, observations, locations, scales) == 0
    @test _AUTHORED_PLATE_SCALE_CALLS[] == length(scales)

    # Broadcast compatibility is established before any pure recipe executes.
    _AUTHORED_PLATE_SCALE_CALLS[] = 0
    @test_throws DimensionMismatch total(
        observations, vec(locations), scales)
    @test _AUTHORED_PLATE_SCALE_CALLS[] == 0

    # A plate is a pure map/reduction contract. Ordinary opaque leaf callables
    # are pure by default; a recipe explicitly known to be effectful is not a
    # plate execution mode and cannot be selected into the scalar plan.
    @test_throws PlanningError (@kernel begin
        scale
        pointwise = plate(scale) do si
            @recipe (effectful = true) y = si + 1
            return y
        end
        return sum(pointwise)
    end)
end

@testset "authored plate block: second transparent object endpoint" begin
    xs = [-0.8, 0.2, 1.1]
    location = -0.1
    scale = 0.9
    reference = [_authored_plate_cauchy(x, location, scale) for x in xs]
    @test authored_cauchy_loglik(xs, location, scale) ≈ sum(reference)
    @test prepare(extract(authored_cauchy_loglik; want = :pointwise))(
        xs, location, scale) ≈ reference
end


@testset "authored plate block: Ref marks an array-valued atom" begin
    observations = [[1.0, -0.5], [0.3, 0.8], [-0.4, 1.2]]
    location = [0.2, -0.1]
    reference = [-0.5 * sum(abs2, x .- location) for x in observations]

    pointwise = prepare(extract(authored_vector_loglik; want = :pointwise))
    total = prepare(authored_vector_loglik)
    untyped_total = prepare(untyped_authored_vector_loglik)
    @test pointwise(observations, location) ≈ reference
    @test total(observations, location) ≈ sum(reference)
    @test untyped_total(observations, location) ≈ sum(reference)
    @test typeof(untyped_total.f).parameters[1] == (1,)
    @test _authored_plate_head_count(code_expr(total), :for) == 1
    @test !occursin("similar", string(code_expr(total)))
end

@testset "authored plate block: derived iterable arguments" begin
    matrix = [1.0 2.0 3.0; 4.0 5.0 6.0]
    column_offsets = [0.25, 0.5, 0.75]
    column_reference = [
        sum(column) + column_offsets[index]
        for (index, column) in enumerate(eachcol(matrix))
    ]

    column_total = prepare(authored_eachcol_sum)
    column_pointwise = prepare(extract(authored_eachcol_sum; want = :pointwise))
    @test column_total(matrix, column_offsets) == sum(column_reference)
    @test column_pointwise(matrix, column_offsets) == column_reference
    @test count(recipe -> recipe.source == :(eachcol(matrix)),
                authored_eachcol_sum.graph.recipes) == 1
    @test length(plan(authored_eachcol_sum).recipes) == 3
    @test _authored_plate_head_count(code_expr(column_total), :for) == 1
    @test !occursin("similar", string(code_expr(column_total)))
    @test !occursin("for ", string(column_total.f.tensorized_ast))

    row_offsets = [0.5, 1.5]
    row_reference = [
        sum(row) + row_offsets[index] + 1.0
        for (index, row) in enumerate(eachrow(matrix))
    ]
    row_total = prepare(authored_eachrow_derived_sum)
    row_pointwise = prepare(extract(authored_eachrow_derived_sum; want = :pointwise))
    @test row_total(matrix, row_offsets) == sum(row_reference)
    @test row_pointwise(matrix, row_offsets) == row_reference
    @test count(recipe -> recipe.source == :(eachrow(matrix)),
                authored_eachrow_derived_sum.graph.recipes) == 1
    @test count(recipe -> recipe.source == :(offsets .+ 1.0),
                authored_eachrow_derived_sum.graph.recipes) == 1
    @test _authored_plate_head_count(code_expr(row_total), :for) == 1
    @test !occursin("for ", string(row_total.f.tensorized_ast))

    values = [-1.0, 0.5, 2.0]
    atomic_offsets = [0.25, 0.75]
    atomic_reference = [
        value + sum(atomic_offsets .+ 1.0) for value in values
    ]
    atomic_total = prepare(authored_ref_derived_sum)
    atomic_pointwise = prepare(extract(authored_ref_derived_sum; want = :pointwise))
    @test atomic_total(values, atomic_offsets) == sum(atomic_reference)
    @test atomic_pointwise(values, atomic_offsets) == atomic_reference
    @test count(recipe -> recipe.source == :(offsets .+ 1.0),
                authored_ref_derived_sum.graph.recipes) == 1
    @test _authored_plate_head_count(code_expr(atomic_total), :for) == 1
    @test !occursin("similar", string(code_expr(atomic_total)))
    @test !occursin("for ", string(atomic_total.f.tensorized_ast))

    logits = [1.0 -0.5 0.25; 0.0 1.25 -0.75; -1.0 0.5 1.5]
    observed = [1, 2, 3]
    categorical_reference = [
        column[observed[index]] - _authored_logsumexp(column)
        for (index, column) in enumerate(eachcol(logits))
    ]
    categorical_total = prepare(authored_eachcol_categorical)
    categorical_pointwise = prepare(extract(
        authored_eachcol_categorical; want = :pointwise))
    @test categorical_total(logits, observed) ≈ sum(categorical_reference)
    @test categorical_pointwise(logits, observed) ≈ categorical_reference
    @test count(recipe -> recipe.source == :(eachcol(logits)),
                authored_eachcol_categorical.graph.recipes) == 1
    @test _authored_plate_head_count(code_expr(categorical_total), :for) == 1
    @test !occursin("similar", string(code_expr(categorical_total)))
    @test !occursin("for ", string(categorical_total.f.tensorized_ast))

    W = [0.25 -0.5; 0.75 1.0]
    b = [-0.25, 0.5]
    coefficients = vcat(vec(W), b)
    normal_reference = sum(
        _authored_plate_normal(value, 0.0, 1.0) for value in coefficients)
    normal_total = prepare(authored_vcat_normal)
    @test normal_total(W, b) ≈ normal_reference
    @test count(recipe -> recipe.source == :(vcat(vec(W), b)),
                authored_vcat_normal.graph.recipes) == 1
    @test _authored_plate_head_count(code_expr(normal_total), :for) == 1
    @test !occursin("similar", string(code_expr(normal_total)))
    @test !occursin("for ", string(normal_total.f.tensorized_ast))
end

# A nested object-endpoint call after a cell-local in a plate cell. The inline
# form was the only working spelling; a typed cell-local collapsed the cell
# return type to Any (`:__return__` vs the endpoint output type), and a keyword
# owner binding could not reference a cell-local. All three now author, agree
# numerically, and stay buffer-free; the log-link keyword route additionally
# drops the object's internal log(scale) round trip.
@kernel authored_plate_endpoint_inline(x, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        authored_normal(li, exp(si)).logpdf(xi)
    end
    return sum(pointwise)
end

@kernel authored_plate_endpoint_typed_local(x, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        log_scale::Float64 = si
        authored_normal(li, exp(log_scale)).logpdf(xi)
    end
    return sum(pointwise)
end

@kernel authored_plate_endpoint_keyword_typed(x, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        log_scale::Float64 = si
        authored_normal(;
            location = li, scale = exp(log_scale), log_scale = log_scale).logpdf(xi)
    end
    return sum(pointwise)
end

@kernel authored_plate_endpoint_keyword_untyped(x, location, scale) = begin
    pointwise = plate(x, location, scale) do xi, li, si
        log_scale = si
        authored_normal(;
            location = li, scale = exp(log_scale), log_scale = log_scale).logpdf(xi)
    end
    return sum(pointwise)
end

# A poisson-like log-link object whose only `log` is the rate round trip (no
# `log(2π)` normalizer to confound the round-trip check below). The `rate`
# HAVE-route lets a caller supply either `rate` or `log_rate`.
@kernel authored_rate_kernel(rate::Float64) = begin
    log_rate::Float64 = log(rate)
    rate::Float64 = exp(log_rate)
    logpdf(k::Float64)::Float64 = k * log_rate - rate
end

@kernel authored_rate_forced(k, lograte) = begin
    pointwise = plate(k, lograte) do ki, lri
        lr = lri
        authored_rate_kernel(exp(lr)).logpdf(ki)
    end
    return sum(pointwise)
end

@kernel authored_rate_direct(k, lograte) = begin
    pointwise = plate(k, lograte) do ki, lri
        lr = lri
        authored_rate_kernel(; log_rate = lr, rate = exp(lr)).logpdf(ki)
    end
    return sum(pointwise)
end

_authored_plate_log_recipe_count(cell) =
    count(recipe -> occursin("log(", string(recipe.source)), cell.recipes)

@testset "authored plate block: cell-local before a nested endpoint call" begin
    x = [0.1, 0.2, -0.3]
    location = [0.0, 0.5, -0.5]
    scale = [0.0, 0.1, -0.1]
    reference = sum(
        _authored_plate_normal(x[i], location[i], exp(scale[i])) for i in eachindex(x))

    for spec in (authored_plate_endpoint_inline,
                 authored_plate_endpoint_typed_local,
                 authored_plate_endpoint_keyword_typed,
                 authored_plate_endpoint_keyword_untyped)
        total = prepare(spec)
        @test total(x, location, scale) ≈ reference
        @test _authored_plate_head_count(code_expr(total), :for) == 1
        @test !occursin("similar", string(code_expr(total)))
    end

    # The keyword log-link route supplies `log_rate` from a cell-local directly,
    # so the cell plan carries no `log(rate)` round trip; the forced `rate = exp`
    # route (also authored via a cell-local here) must still recompute it.
    k = [3.0, 5.0, 2.0]
    lograte = [-0.5, 0.25, 1.0]
    @test authored_rate_direct(k, lograte) ≈ authored_rate_forced(k, lograte)
    forced_cell = plate_body(first(plan(authored_rate_forced).recipes))
    direct_cell = plate_body(first(plan(authored_rate_direct).recipes))
    @test _authored_plate_log_recipe_count(forced_cell) == 1
    @test _authored_plate_log_recipe_count(direct_cell) == 0
    @test !occursin("similar", string(code_expr(prepare(authored_rate_direct))))

    # A genuine undeclared owner port must still fail loudly, not be swept in as a
    # cell-local materialization.
    @test_throws ArgumentError @macroexpand @kernel authored_plate_endpoint_bad(
            x, location, scale) = begin
        pointwise = plate(x, location, scale) do xi, li, si
            authored_normal(;
                location = li, scale = exp(si), log_scale = missing_port).logpdf(xi)
        end
        return sum(pointwise)
    end
end

@testset "authored plate marker: untyped scalar leading arg (want=:pointwise)" begin
    # Regression guard (source-review catch by ReactiveKernels:performance): the
    # static axis-marker fast path must NOT select an untyped (metadata-`Any`)
    # leading argument that holds a runtime SCALAR. `want=:pointwise` materializes
    # `_plate_similar_output(marker, ...)`, so a wrong marker builds the pointwise buffer from a
    # scalar (or errors). Here `plate(x, location, scale)` is called with a scalar
    # `x` and a vector `location`; `_authored_plate_is_axis` skips `x` and selects
    # `location`, so the static classifier — for which an `Any` port is
    # `:ambiguous` — must fall back to the runtime marker and reach the same axis.
    # This is the correctness companion to the `test_ad.jl` Enzyme regression: the
    # M0-shaped case proves the fast path lowers, this proves it never fires when
    # it cannot prove the axis.
    pointwise = prepare(extract(untyped_authored_normal_loglik; want = :pointwise))
    x = 0.25
    location = [0.1, 0.2, 0.4, 0.5]
    scale = 0.8
    result = pointwise(x, location, scale)
    reference = [-0.5 * log(2π) - 0.5 * ((x - location[i]) / scale)^2 - log(scale)
                 for i in eachindex(location)]
    @test result isa AbstractVector
    @test length(result) == length(location)
    @test result ≈ reference
end

module TupleAxisNative
using ReactiveKernels

@kernel broadcast_axes(q::Vector{Float64}, x, y) = begin
    pointwise = plate(x, Ref(q), y) do xi, whole, yi
        xi + sum(whole) * yi
    end
    total::Float64 = sum(pointwise)
    return total
end

# A plate whose ONLY batched axis is a tuple (no array co-operand).
@kernel tuple_only(q::Vector{Float64}, x) = begin
    pointwise = plate(x, Ref(q)) do xi, whole
        xi + sum(whole)
    end
    total::Float64 = sum(pointwise)
    return total
end
end

@testset "authored plate native: tuple / singleton axis pointwise output" begin
    # Regression (todo `native-tuple-axi`, reporter ReactiveKernels:performance):
    # a `Tuple` is admitted as a batched axis — `_static_plate_axis_class(
    # ::Type{<:Tuple}) === :axis` and `_authored_plate_is_axis(::Tuple)` accept it —
    # so the static marker binds to a tuple argument. The pointwise buffer was then
    # allocated with a bare `similar(marker, T, axes)`, which has NO method for a
    # `Tuple` marker: the Reactant path materialized the tuple case
    # (test_ref_array_plate_reactant.jl "broadcast axes and shared rank …") but the
    # NATIVE `k(q)` crashed with
    # `similar(::Tuple{Float64}, ::Type{Float64}, ::Tuple{Base.OneTo{Int}})`.
    # `_plate_similar_output` allocates a plain `Array` for a tuple marker, matching
    # Julia's broadcast container rule (`x .+ sum(q) .* y` is a `Vector`).
    q = [0.7, -0.3]

    # A tuple axis beside a Vector axis, both `want`s.
    for (x, y) in (((1.0,), collect(1.0:32)),        # singleton tuple, real broadcast
                   ((1.0, 2.0, 3.0), fill(0.5, 3)))  # multi-element tuple
        k = prepare(TupleAxisNative.broadcast_axes; want = :pointwise, bound = (; x, y))
        result = k(q)
        reference = x .+ sum(q) .* y
        @test result isa Vector{Float64}
        @test size(result) == size(reference)
        @test result ≈ reference
        # The total path (no pointwise buffer) already worked; keep it in parity.
        ktot = prepare(TupleAxisNative.broadcast_axes; want = :total, bound = (; x, y))
        @test ktot(q) ≈ sum(reference)
    end

    # A plate whose SOLE axis is a tuple. `_plate_similar`'s style-combine cannot
    # cover this — a lone `Style{Tuple}` materializes a tuple, not an array — so it
    # exercises the `_plate_similar_output(::Tuple, …)` marker dispatch directly.
    x = (1.0, 2.0, 3.0, 4.0)
    k = prepare(TupleAxisNative.tuple_only; want = :pointwise, bound = (; x))
    result = k(q)
    reference = collect(x) .+ sum(q)
    @test result isa Vector{Float64}
    @test result ≈ reference
end

module UninferredPlateCells
using ReactiveKernels

# Hides the cell's result type from inference, as Julia's inference can for a
# nested prepared kernel called in a lazy arm (`_declare_typed_output!`).
opaque(x) = Base.inferencebarrier(x)

@kernel group_total(groups) = begin
    cells = plate(groups) do xs
        opaque(2.0 * sum(xs))
    end
    total = sum(cells)
    return total
end

@kernel indexed_total(idx, values) = begin
    cells = plate(idx, Ref(values)) do i, v
        opaque(v[i])
    end
    total = sum(cells)
    return total
end

@kernel scan_total(xs) = begin
    trajectory = scan(xs; init = 0.0) do carry, x
        next = carry + x
        (next, next)
    end
    cells = plate(trajectory) do t
        opaque(2.0 * t)
    end
    total = sum(cells)
    return total
end
end

@testset "authored plate total: cells whose result type does not infer" begin
    # Regression (todo `1a2te8d`): the fused total of a cell inferred as `Any`
    # was seeded with `zero(eltype(axis))`, which fails for an axis of vectors
    # with `zero(::Type{Vector{Float64}})`. The first cell now starts the total.
    U = UninferredPlateCells
    groups = prepare(U.group_total)
    @test groups([[1.0, 2.0], [0.5], Float64[]]) === 7.0
    # refused: like Base's `sum(Any[])`, an empty sum of cells without an
    # inferred element type has no zero; the error names the declaration.
    @test_throws "Declare the cell's result type" groups(Vector{Float64}[])

    # An empty numeric axis keeps its element type's zero.
    indexed = prepare(U.indexed_total)
    @test indexed([1, 3], [0.5, 1.0, 2.0]) === 2.5
    @test indexed(Int[], [0.5]) === 0

    # The same seed when a summed plate consumes a scan in its loop.
    scanned = prepare(U.scan_total)
    @test !occursin("trajectory", sprint(show, readable_code(scanned)))
    @test scanned([1.0, 2.0]) === 8.0
    @test scanned(Float64[]) === 0.0
end

@testset "authored plate chain: redundant axis-check elision + preserved domain guards" begin
    # snag composed-authore: a fused authored plate chain absorbed one axis-check
    # group per sub-plate, so it emitted redundant pre-loop
    # `_plate_require_axes(combine_axes(...))` calls an equivalent single plate
    # never has (the only codegen difference; reported to slow the primal under
    # concurrent load, but NOT reproducible under controlled conditions —
    # structural cleanup, not a proven speedup). These lock BOTH halves of it:
    # the redundant checks are ELIDED over typed/bound array ports, and the
    # axis-domain guards are PRESERVED (an axis-less producer sub-plate must still
    # be rejected, never silently borrow a sibling's axis).

    # POSITIVE — check elision + value: a typed fused chain lowers to the SAME
    # number of axis checks as the equivalent single plate, and the same value.
    @kernel _ce_chain(q::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64}) = begin
        location::Float64 = q[1]
        slope::Float64 = q[2]
        middle = plate(x, location) do xi, loc
            2.0 * xi + loc
        end
        pointwise = plate(y, middle, slope) do yi, mi, sl
            yi + mi^2 + sl * mi
        end
        total::Float64 = sum(pointwise)
    end
    @kernel _ce_oneplate(q::Vector{Float64}, x::Vector{Float64}, y::Vector{Float64}) = begin
        location::Float64 = q[1]
        slope::Float64 = q[2]
        pointwise = plate(x, y, location, slope) do xi, yi, loc, sl
            mi = 2.0 * xi + loc
            yi + mi^2 + sl * mi
        end
        total::Float64 = sum(pointwise)
    end
    q = [0.3, -0.4]
    x = collect(range(-1.0, 1.0; length = 16))
    y = collect(range(0.25, -0.25; length = 16))
    ch = prepare(_ce_chain; have = (:q, :x, :y), want = :total, bound = (; x, y))
    op = prepare(_ce_oneplate; have = (:q, :x, :y), want = :total, bound = (; x, y))
    @test ch(q) ≈ op(q)
    axis_checks(k) = count("_plate_require_axes", string(code_expr(k)))
    @test axis_checks(ch) == axis_checks(op)   # redundant per-sub-plate checks elided
    @test axis_checks(ch) == 1

    # NEGATIVE — an axis-less producer sub-plate in a fused chain must STILL be
    # rejected (it has no batched axis of its own). The reviewed regression let an
    # EMPTY-filtered axis-check group be skipped, so the producer silently borrowed
    # the consumer's axis and returned a wrong value instead of throwing. These
    # match the pre-fusion / single-plate rejection (a zero-operand
    # `combine_axes()` — currently a `MethodError`).
    @kernel _ce_scalar_producer(s::Float64, y::Vector{Float64}) = begin
        producer = plate(s) do si
            si * 2.0
        end
        pointwise = plate(y, producer) do yi, pj
            yi + pj
        end
        total::Float64 = sum(pointwise)
    end
    ksp = prepare(_ce_scalar_producer)
    @test_throws MethodError ksp(2.0, y)

    @kernel _ce_ref_producer(a::Float64, y::Vector{Float64}) = begin
        producer = plate(Ref(a)) do ra
            ra * 2.0
        end
        pointwise = plate(y, producer) do yi, pj
            yi + pj
        end
        total::Float64 = sum(pointwise)
    end
    krp = prepare(_ce_ref_producer)
    @test_throws MethodError krp(2.0, y)
end

@testset "authored plate block: automatic caller-scalar threading" begin
    @kernel _free_scalar_plate(
            y::Vector{Float64}, mu::Vector{Float64}, s::Float64) = begin
        pointwise = plate(y, mu) do yi, mui
            (mui + s)^2 + yi
        end
        total::Float64 = sum(pointwise)
    end

    y = [0.2, -0.4, 1.1]
    mu = [0.0, 0.3, -0.2]
    kernel = prepare(_free_scalar_plate; want = :total)
    @test kernel(y, mu, 0.7) ≈ sum(((m + 0.7)^2 + obs)
                                   for (obs, m) in zip(y, mu))
end

@testset "authored plate block: keyword-call authoring without semicolon" begin
    scale_logpdf(; logit) = -log1pexp(-logit)
    @kernel _kw_call_plate(e::Vector{Float64}) = begin
        pointwise = plate(e) do ei
            scale_logpdf(logit = ei) * ei
        end
        total::Float64 = sum(pointwise)
    end

    e = [-1.0, 0.0, 0.25, 2.0]
    expected = map(e) do x
        -log1pexp(-x) * x
    end
    @test prepare(_kw_call_plate; want = :pointwise)(e) ≈ expected
end

# A plate cell whose fused closure carries a LOOP — a `sum(generator)` over an
# inner host axis, the dose-superposition shape — was a real function call on
# every plate coordinate on Julia 1.10: the inlining heuristic refuses a
# loop-carrying closure, so the cell's loop invariants (`plan.shifts`,
# `eachindex`) were recomputed per observation, at 2× the time of the same
# body written inline. `_kernel_source_call` now callsite-inlines the native
# fused closure (snag `generator-plate-e160f4c2`). Julia 1.12 inlines this
# closure on its own, so the plain-call control is asserted only where the
# heuristic refuses it.
@testset "authored plate cell: a loop-carrying cell closure is inlined into the native loop" begin
    plan = (; shifts = [0, 4, 9], nobs = 4096)
    _slots(plan) = eachindex(plan.shifts)
    _lookup(observation, plan, i) = observation - plan.shifts[i]
    @kernel _generator_cell(plan, units, weights) = begin
        observations::UnitRange{Int} = 1:plan.nobs
        concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units),
                                               Ref(weights)) do observation, schedule_plan, response, amounts
            sum((ifelse(_lookup(observation, schedule_plan, i) > 0,
                        response[max(_lookup(observation, schedule_plan, i), 1)] * amounts[i],
                        0.0)
                 for i in _slots(schedule_plan)); init = 0.0)
        end
        return concentration
    end
    units = [sin(0.01k) + 1.5 for k in 1:plan.nobs]
    weights = [0.5, 1.25, 0.75]
    kernel = prepare(_generator_cell)
    reference = [sum(observation - s > 0 ? units[observation - s] * w : 0.0
                     for (s, w) in zip(plan.shifts, weights))
                 for observation in 1:plan.nobs]
    @test kernel(plan, units, weights) ≈ reference
    # Output-only allocation: a per-cell allocation would add ≥ 32 B per observation.
    kernel(plan, units, weights)
    @test (@allocated kernel(plan, units, weights)) < 3 * sizeof(Float64) * plan.nobs

    # The cell op is the fused closure whose recipe yields the plate value; its
    # closure arguments are the do-block formals in recipe-input order.
    cell_index = only(i for (i, recipe) in enumerate(kernel.lowered_recipes)
                      if [v.name for v in recipe.outputs] == [:__plate_value__])
    op = kernel.ops[cell_index]
    @test op isa ReactiveKernels._KernelSourceOp
    formal_types = Dict(:schedule_plan => typeof(plan), :observation => Int,
                        :response => typeof(units), :amounts => typeof(weights))
    argtypes = Tuple(formal_types[v.name] for v in kernel.lowered_recipes[cell_index].inputs)
    invokes_closure(f, types) = any(only(Base.code_typed(f, types; optimize = true))[1].code) do stmt
        stmt isa Expr && stmt.head === :invoke || return false
        callee = stmt.args[1]
        callee isa Core.CodeInstance && (callee = callee.def)
        callee isa Core.MethodInstance || return false
        # The specialized signature names the concrete closure type (a cell
        # closure over local helpers is parametric; the method's own `sig` is not).
        callee.specTypes isa DataType && callee.specTypes.parameters[1] in
            (typeof(op.f), typeof(ReactiveKernels._kernel_native_source(op.f)))
    end
    @test !invokes_closure(ReactiveKernels._kernel_source_call,
                           (Val{:native}, typeof(op), argtypes...))
    # Negative control: explicitly keep the same source callable as an invoke,
    # independently of the Julia version's ordinary inlining heuristic.
    # Fixed arity on purpose: a forwarded `args...` splat is left unspecialized
    # and lowers to a dynamic apply, which the `invoke` scan cannot see.
    retained_call(op, a, b, c, d) = Base.@noinline op.f(a, b, c, d)
    @test length(argtypes) == 4
    @test invokes_closure(retained_call, (typeof(op), argtypes...))
end

# --- RK's own plate reads and stores carry no bounds checks (snag native-lowering-948c4ab6)
# The native plate loop reads each cell argument at a coordinate of
# `CartesianIndices(combine_axes(arguments))` and stores the cell value at the
# same coordinate of a buffer with those axes: both are in bounds by
# construction, as in Base's broadcast `copyto!`. Checked, they kept LLVM from
# vectorizing even an arithmetic cell; a fixed-degree polynomial cell with tuple
# coefficients ran 2.6x its hand loop. The cell body keeps its own checks.
import InteractiveUtils
@kernel _unchecked_affine_plate(xs::Vector{Float64}) = begin
    ys::Vector{Float64} = plate(xs) do x
        muladd(x, 2.0, 1.0)
    end
    return ys
end
@kernel _unchecked_poly_plate(xs::Vector{Float64}, c) = begin
    ys::Vector{Float64} = plate(xs, Ref(c)) do x, c
        evalpoly(x, c)
    end
    return ys
end
@kernel _unchecked_broadcast_plate(a::Matrix{Float64}, b::Matrix{Float64}) = begin
    ys = plate(a, b) do x, y
        x - 2y
    end
    return ys
end
@kernel _checked_cell_plate(xs::Vector{Float64}, idx::Vector{Int}, table::Vector{Float64}) = begin
    ys::Vector{Float64} = plate(xs, idx, Ref(table)) do x, i, t
        x * t[i]
    end
    return ys
end

function _plate_entry_llvm(kernel, args...)
    kernel(args...)
    sprint(io -> InteractiveUtils.code_llvm(io,
        ReactiveKernels.RuntimeGeneratedFunctions.generated_callfunc,
        Tuple{typeof(kernel.f.native), typeof(kernel.ops), map(typeof, args)...};
        debuginfo = :none))
end

@testset "authored plate native: RK's own cell reads and stores are unchecked" begin
    xs = [1 / (k + 0.5) for k in 0:999]
    coefficients = [1.0, -0.25, 0.125, 0.3, -0.07, 0.011]
    affine = prepare(_unchecked_affine_plate)
    poly = prepare(_unchecked_poly_plate)
    @test affine(xs) == muladd.(xs, 2.0, 1.0)
    # `muladd` leaves contraction to the compiler, so compare to rounding.
    @test poly(xs, Tuple(coefficients)) ≈ evalpoly.(xs, Ref(Tuple(coefficients))) rtol = 1e-14
    @test poly(xs, coefficients) ≈ evalpoly.(xs, Ref(coefficients)) rtol = 1e-14
    @test affine(Float64[]) == Float64[]

    # Singleton expansion reads through Base's extruded projection unchanged.
    broadcast_plate = prepare(_unchecked_broadcast_plate)
    a, b = reshape([0.5, 1.5, -2.0], 3, 1), reshape([1.0, 2.0, 3.0, 4.0], 1, 4)
    @test broadcast_plate(a, b) == a .- 2 .* b
    @test broadcast_plate(a, repeat(b, 3)) == a .- 2 .* b

    # Neither the plate's projection nor its store is checked, so the cell loop
    # vectorizes like the hand loop.
    for (kernel, args) in ((affine, (xs,)), (poly, (xs, Tuple(coefficients))))
        llvm = _plate_entry_llvm(kernel, args...)
        @test !occursin("bounds_error", llvm)
        if Sys.ARCH in (:x86_64, :aarch64)
            @test occursin(r"<\d+ x double>", llvm)
        end
    end

    # Control: the cell body's own indexing keeps its bounds check.
    checked = prepare(_checked_cell_plate)
    table = [2.0, 3.0, 5.0]
    @test checked([1.0, 2.0, 3.0], [3, 1, 2], table) == [5.0, 4.0, 9.0]
    @test occursin("bounds_error", _plate_entry_llvm(checked, [1.0], [1], table))
    @test_throws BoundsError checked([1.0, 2.0], [1, 4], table)
end

# --- per-cell reductions over a few host indices (snag plate-cell-gathe-94d4a929)
# A plate cell that superposes a few dose responses per observation. The
# reporter authored it as vector temporaries (`observation .- shifts`,
# `max.(row, 1)`, a gather, a mask, a fused product, `sum`): every temporary is
# an ordinary per-cell Julia allocation, so the prepared plate allocated ~336 B
# per cell and ran ~50× slower than a hand loop. A scalar generator over the
# per-dose index is allocation-free natively AND lowers per lane under Reactant
# (`test_ref_array_plate_reactant.jl`), with either plan representation behind a
# per-dose accessor: index arithmetic on host data for the lattice plan, and the
# one-element reduction `sum(view(row, i:i))` for a stored row (the same
# normalization the tensorized `row[i]` lowering uses, so it never trips
# Reactant's scalar-indexing guard inside an opaque helper).
struct _PlateGatherLattice
    shifts::Vector{Int}
    nobs::Int
end
struct _PlateGatherExact
    rows::Matrix{Int}
end
_plate_gather_domain(plan::_PlateGatherLattice) = collect(1:plan.nobs)
_plate_gather_domain(plan::_PlateGatherExact) = eachrow(plan.rows)
_plate_gather_slots(plan::_PlateGatherLattice) = eachindex(plan.shifts)
_plate_gather_slots(plan::_PlateGatherExact) = axes(plan.rows, 2)
_plate_gather_index(observation, plan::_PlateGatherLattice, i) =
    observation - plan.shifts[i]
_plate_gather_index(row, ::_PlateGatherExact, i) = sum(view(row, i:i))

@kernel authored_gather_generator_plate(plan, units, weights) = begin
    observations = _plate_gather_domain(plan)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do observation, schedule_plan, response, amounts
        sum(ifelse(_plate_gather_index(observation, schedule_plan, i) > 0,
                   response[max(_plate_gather_index(observation, schedule_plan, i), 1)] *
                       amounts[i], 0.0)
            for i in _plate_gather_slots(schedule_plan))
    end
    return concentration
end

# The reporter's vector-temporary cell, kept as the value reference and as the
# allocation control this testset contrasts against.
_plate_gather_row(row, response, amounts) =
    sum(response[max.(row, 1)] .* (row .> 0) .* amounts)
@kernel authored_gather_broadcast_plate(plan, units, weights) = begin
    observations = collect(1:plan.nobs)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do observation, schedule_plan, response, amounts
        row = observation .- schedule_plan.shifts
        _plate_gather_row(row, response, amounts)
    end
    return concentration
end

# Function barrier so `@allocated` measures the kernel, not global-ref boxing.
_plate_gather_allocated(k, plan, units, weights) = @allocated k(plan, units, weights)

@testset "authored plate block: per-cell generator reduction over host indices" begin
    shifts = [0, 40, 100]
    weights = [3.0, 1.5, 0.25]
    generator = prepare(authored_gather_generator_plate)
    broadcast_cell = prepare(authored_gather_broadcast_plate)
    # Bytes per call at two lane counts: the growth between them is the
    # per-cell cost, independent of the fixed call overhead.
    bytes = Dict{Tuple{Symbol,Int},Int}()
    for nobs in (257, 513)
        lattice = _PlateGatherLattice(shifts, nobs)
        exact = _PlateGatherExact([o - s for o in 1:nobs, s in shifts])
        units = collect(range(0.5, 2.0; length = nobs))
        expected = [sum(o - s > 0 ? units[o - s] * w : 0.0
                        for (s, w) in zip(shifts, weights)) for o in 1:nobs]
        @test generator(lattice, units, weights) ≈ expected
        @test generator(exact, units, weights) ≈ expected
        @test broadcast_cell(lattice, units, weights) ≈ expected
        _plate_gather_allocated(generator, lattice, units, weights)
        _plate_gather_allocated(generator, exact, units, weights)
        _plate_gather_allocated(broadcast_cell, lattice, units, weights)
        bytes[(:lattice, nobs)] = _plate_gather_allocated(generator, lattice, units, weights)
        bytes[(:exact, nobs)] = _plate_gather_allocated(generator, exact, units, weights)
        bytes[(:broadcast, nobs)] = _plate_gather_allocated(broadcast_cell, lattice, units, weights)
    end
    growth(kind) = bytes[(kind, 513)] - bytes[(kind, 257)]
    per_cell_output = sizeof(Float64) * (513 - 257)
    # Lattice: the output plus the `collect(1:nobs)` domain recipe grow with
    # the lane count; exact rows (`eachrow` is lazy): the output only. Nothing
    # per cell in either case.
    @test growth(:lattice) <= 2 * per_cell_output + 64
    @test growth(:exact) <= per_cell_output + 64
    # The vector-temporary cell allocates several small arrays per cell.
    @test growth(:broadcast) > 100 * (513 - 257)
end

# --- the natural superposition cell (snag one-natural-supe-39da86a4)
# The concentration at observation t is the sum, over the doses already given,
# of the dose weight times the unit response at the lag since that dose. Two
# natural spellings state exactly that: the unit response extended by zero
# before its dose (`get(units, lag, 0.0)`, Base's total gather), and the sum
# over the doses given (a filtered generator). Both are ordinary Julia natively;
# the tensorized companion keeps each branch lazy (`test_ref_array_plate_reactant.jl`).
# One graph serves both plan types through dispatching helpers: its domain port
# carries no declared type, and the native lowering schedules it from the
# concrete value it receives.
struct _NaturalSupLattice
    shifts::Vector{Int}
    nobs::Int
end
struct _NaturalSupExact
    rows::Matrix{Int}
end
_natural_sup_domain(plan::_NaturalSupLattice) = 1:plan.nobs
_natural_sup_domain(plan::_NaturalSupExact) = eachrow(plan.rows)
_natural_sup_doses(plan::_NaturalSupLattice) = eachindex(plan.shifts)
_natural_sup_doses(plan::_NaturalSupExact) = axes(plan.rows, 2)
_natural_sup_lag(t, plan::_NaturalSupLattice, j) = t - plan.shifts[j]
_natural_sup_lag(row, ::_NaturalSupExact, j) = row[j]

@kernel natural_sup_get(plan, units, weights) = begin
    observations::UnitRange{Int} = 1:plan.nobs
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum(w[j] * get(u, t - p.shifts[j], 0.0) for j in eachindex(p.shifts); init = 0.0)
    end
    return concentration
end
@kernel natural_sup_filter(plan, units, weights) = begin
    observations::UnitRange{Int} = 1:plan.nobs
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum(w[j] * u[t - p.shifts[j]] for j in eachindex(p.shifts) if t > p.shifts[j]; init = 0.0)
    end
    return concentration
end
@kernel natural_sup_one_graph(plan, units, weights) = begin
    observations = _natural_sup_domain(plan)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum(w[j] * get(u, _natural_sup_lag(t, p, j), 0.0) for j in _natural_sup_doses(p); init = 0.0)
    end
    return concentration
end
# The same one-graph cell with the domain port declared, the per-plan-type
# split this graph replaces; the allocation control below.
@kernel natural_sup_lattice_typed(plan, units, weights) = begin
    observations::UnitRange{Int} = 1:plan.nobs
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum(w[j] * get(u, _natural_sup_lag(t, p, j), 0.0) for j in _natural_sup_doses(p); init = 0.0)
    end
    return concentration
end

_natural_sup_allocated(k, plan, units, weights) = @allocated k(plan, units, weights)
_natural_sup_slots(s) = eachindex(s)
# Slack for the cross-kernel allocation comparisons below. Exact byte identity
# between differently-lowered kernels is not portable: Windows Julia 1.13.1
# measured +32/+48 B for the undeclared-domain kernel against the declared one
# (CI run 36811107488; the declared baseline itself reads 2183 B there against
# 2184 B on Linux), while both execute identical allocation sequences on Linux.
# The slack absorbs such platform constants while keeping the documented
# regressions detectable: the 304 B runtime marker and any per-cell allocation
# (at least 16 B per observation here).
_natural_sup_parity_margin = 128

@testset "authored plate block: natural superposition cell (get, filtered sum, one graph)" begin
    nobs = 257
    units = collect(range(0.5, 2.0; length = nobs))
    kernels = map(prepare, (natural_sup_get, natural_sup_filter, natural_sup_one_graph,
                            natural_sup_lattice_typed))
    get_cell, filter_cell, one_graph, typed = kernels
    for shifts in ([0, 40, 100], [3, 3, 250], Int[])
        weights = collect(range(0.75, 1.5; length = length(shifts)))
        expected = [sum((t > s ? w * units[t - s] : 0.0 for (s, w) in zip(shifts, weights));
                        init = 0.0) for t in 1:nobs]
        lattice = _NaturalSupLattice(shifts, nobs)
        exact = _NaturalSupExact([max(t - s, 0) for t in 1:nobs, s in shifts])
        @test get_cell(lattice, units, weights) == expected
        @test filter_cell(lattice, units, weights) == expected
        @test one_graph(lattice, units, weights) == expected
        @test one_graph(exact, units, weights) == expected
        @test typed(lattice, units, weights) == expected
    end

    # The undeclared domain schedules like the declared one: the output is the
    # only allocation, on both plan types (before, the runtime axis marker
    # allocated 304 B per call and every cell carried the axis-changed guard).
    shifts = [0, 40, 100]
    weights = [3.0, 1.5, 0.25]
    lattice = _NaturalSupLattice(shifts, nobs)
    exact = _NaturalSupExact([max(t - s, 0) for t in 1:nobs, s in shifts])
    for (k, plan) in ((one_graph, lattice), (one_graph, exact), (typed, lattice))
        _natural_sup_allocated(k, plan, units, weights)
    end
    output_bytes = _natural_sup_allocated(typed, lattice, units, weights)
    # A per-cell allocation would add at least 16 B per observation.
    @test output_bytes < 2 * sizeof(Float64) * nobs
    # Bounded parity (slack above): the undeclared domain adds no per-call
    # scheduling allocation on either plan type.
    @test _natural_sup_allocated(one_graph, lattice, units, weights) <=
          output_bytes + _natural_sup_parity_margin
    @test _natural_sup_allocated(one_graph, exact, units, weights) <=
          output_bytes + _natural_sup_parity_margin
end

# Dose-outer lowering of a gathered generator sum (`_KernelReduction`). The
# control reads through an alias of `get` that the lowering does not recognize,
# so it keeps the observation-outer cell loop; every value must match it
# bitwise, including the NaN an infinite weight makes outside the window.
const _dose_outer_fetch = Base.get
@kernel dose_outer_cell(plan, units, weights) = begin
    observations = _natural_sup_domain(plan)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum(w[j] * get(u, _natural_sup_lag(t, p, j), 0.0) for j in eachindex(w); init = 0.0)
    end
    total = sum(concentration)
    return concentration, total
end
@kernel dose_outer_control(plan, units, weights) = begin
    observations = _natural_sup_domain(plan)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum(w[j] * _dose_outer_fetch(u, _natural_sup_lag(t, p, j), 0.0) for j in eachindex(w); init = 0.0)
    end
    total = sum(concentration)
    return concentration, total
end
# An `Int` seed against `Float64` terms changes the accumulator type after the
# first dose: the lowering keeps the cell loop.
@kernel dose_outer_int_seed(plan, units, weights) = begin
    observations = _natural_sup_domain(plan)
    concentration = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum(w[j] * get(u, _natural_sup_lag(t, p, j), 0.0) for j in eachindex(w); init = 0)
    end
    return concentration
end
# A dose loop whose range reads the cell is not a plate invariant.
@kernel dose_outer_cell_range(plan, units, weights) = begin
    observations = _natural_sup_domain(plan)
    concentration::Vector{Float64} = plate(observations, Ref(plan), Ref(units), Ref(weights)) do t, p, u, w
        sum(w[j] * get(u, _natural_sup_lag(t, p, j), 0.0) for j in 1:min(t, length(w)); init = 0.0)
    end
    return concentration
end
_dose_outer_reductions(kernel) = count(op -> op isa ReactiveKernels._KernelSourceOp &&
                                             op.f isa ReactiveKernels._KernelReduction, kernel.ops)
_dose_outer_same(a, b) = length(a) == length(b) && all(map(===, a, b))

@testset "authored plate block: a gathered generator sum runs dose-outer, bitwise" begin
    cell, control = prepare(dose_outer_cell), prepare(dose_outer_control)
    @test _dose_outer_reductions(cell) == 1
    @test _dose_outer_reductions(control) == 0
    # The filtered sum skips the term (and its index): no dose-outer parts.
    @test _dose_outer_reductions(prepare(natural_sup_filter)) == 0
    nobs = 257
    for (shifts, weights, nunits) in (
            ([0, 40, 100], [3.0, 1.5, 0.25], nobs),
            ([3, 3, 250], [0.75, -1.0, 2.0], nobs),
            ([300, 0], [1.0, 2.0], nobs),               # a dose after the horizon
            ([-5, 10], [1.0, 2.0], nobs),               # lags past the response
            ([0, 40], [-0.0, Inf], nobs),               # NaN outside the window
            ([0, 40, 100], [3.0, 1.5, 0.25], 50),       # a shorter response
            ([0, 40], [1.0, 2.0], 0),                   # an empty response
            (Int[], Float64[], nobs))
        units = nunits == 0 ? Float64[] : collect(range(0.5, 2.0; length = nunits))
        lattice = _NaturalSupLattice(shifts, nobs)
        exact = _NaturalSupExact([max(t - s, 0) for t in 1:nobs, s in shifts])
        for plan in (lattice, exact)
            got, expected = cell(plan, units, weights), control(plan, units, weights)
            @test _dose_outer_same(got[1], expected[1])
            @test got[2] === expected[2]
        end
    end
    # The authored values, for the plain cases.
    units = collect(range(0.5, 2.0; length = nobs))
    shifts, weights = [3, 3, 250], [0.75, -1.0, 2.0]
    @test cell(_NaturalSupLattice(shifts, nobs), units, weights)[1] ==
          [sum((t > s ? w * units[t - s] : 0.0 for (s, w) in zip(shifts, weights)); init = 0.0)
           for t in 1:nobs]
    # An empty plate.
    @test cell(_NaturalSupLattice([0, 4], 0), units, [1.0, 2.0]) == (Float64[], 0.0)
    # Not split, same values: a non-`Vector` response, an `Int` seed, a dose
    # range that reads the cell.
    lattice = _NaturalSupLattice([0, 40, 100], nobs)
    weights = [3.0, 1.5, 0.25]
    @test _dose_outer_same(cell(lattice, view(units, :), weights)[1],
                           control(lattice, view(units, :), weights)[1])
    int_seed = prepare(dose_outer_int_seed)
    @test _dose_outer_reductions(int_seed) == 1
    @test int_seed(lattice, units, weights) == control(lattice, units, weights)[1]
    @test _dose_outer_reductions(prepare(dose_outer_cell_range)) == 1
    @test prepare(dose_outer_cell_range)(lattice, units, weights) ==
          [sum((w * get(units, t - s, 0.0) for (s, w) in
                Iterators.take(zip(lattice.shifts, weights), min(t, 3))); init = 0.0)
           for t in 1:nobs]
    # The output is the only allocation, as for the cell loop (bounded parity,
    # same slack: the two lowerings differ, so exact identity is not portable).
    _natural_sup_allocated(cell, lattice, units, weights)
    _natural_sup_allocated(control, lattice, units, weights)
    @test _natural_sup_allocated(cell, lattice, units, weights) <=
          _natural_sup_allocated(control, lattice, units, weights) +
          _natural_sup_parity_margin
end

# A domain with two axes runs the same passes over its Cartesian cells
# (`_plate_cells`), in coordinate order.
@kernel dose_outer_grid(observations, shifts, units, weights) = begin
    concentration = plate(observations, Ref(shifts), Ref(units), Ref(weights)) do t, s, u, w
        sum(w[j] * get(u, t - s[j], 0.0) for j in eachindex(w); init = 0.0)
    end
    total = sum(concentration)
    return concentration, total
end
@kernel dose_outer_grid_control(observations, shifts, units, weights) = begin
    concentration = plate(observations, Ref(shifts), Ref(units), Ref(weights)) do t, s, u, w
        sum(w[j] * _dose_outer_fetch(u, t - s[j], 0.0) for j in eachindex(w); init = 0.0)
    end
    total = sum(concentration)
    return concentration, total
end

@testset "authored plate block: dose-outer over a two-axis domain" begin
    grid, control = prepare(dose_outer_grid), prepare(dose_outer_grid_control)
    @test _dose_outer_reductions(grid) == 1
    units = collect(range(0.5, 2.0; length = 200))
    for (shifts, weights) in (([0, 40, 100], [3.0, 1.5, 0.25]), ([0, 40], [-0.0, Inf]))
        observations = reshape(collect(1:255), 15, 17)
        got, expected = grid(observations, shifts, units, weights),
                        control(observations, shifts, units, weights)
        @test size(got[1]) == (15, 17)
        @test _dose_outer_same(vec(got[1]), vec(expected[1]))
        @test got[2] === expected[2]
    end
end

# Coefficient-outer lowering of an `evalpoly(x, c)` cell over shared
# coefficients (snag native-lowering-948c4ab6). Base's `evalpoly` over a vector
# is a per-cell runtime Horner loop; the lowering runs each coefficient as one
# pass over a tile of cells, with Base's operations in Base's order. The
# control calls the same `evalpoly` through a function the lowering does not
# recognize, so it keeps the cell loop.
_plate_horner(x, c) = evalpoly(x, c)
@kernel evalpoly_cell(xs, c) = begin
    ys = plate(xs, Ref(c)) do x, c
        evalpoly(x, c)
    end
    total = sum(ys)
    return ys, total
end
@kernel evalpoly_cell_reordered(xs, c) = begin
    ys = plate(Ref(c), xs) do c, x
        evalpoly(x, c)
    end
    return ys
end
@kernel evalpoly_control(xs, c) = begin
    ys = plate(xs, Ref(c)) do x, c
        _plate_horner(x, c)
    end
    total = sum(ys)
    return ys, total
end
_evalpoly_lowered(kernel) = occursin("_plate_evalpoly_ready", string(code_expr(kernel)))

@testset "authored plate block: an evalpoly cell over shared coefficients runs coefficient-outer" begin
    cell, control = prepare(evalpoly_cell), prepare(evalpoly_control)
    @test _evalpoly_lowered(cell)
    @test !_evalpoly_lowered(control)
    @test _evalpoly_lowered(prepare(evalpoly_cell_reordered))
    coefficients = [1.0, -0.25, 0.125, 0.3, -0.07, 0.011, 2.5e-3, -4.0e-4, 1.0e-5, 3.0e-6, -7.0e-7]
    # Lengths around the tile, and degrees down to a constant.
    for n in (0, 1, 255, 256, 257, 513, 1000), c in (coefficients, coefficients[1:2], [2.5])
        xs = [1 / (k + 0.5) for k in 0:n-1]
        got, expected = cell(xs, c), control(xs, c)
        @test got[1] == expected[1] == evalpoly.(xs, Ref(c))
        @test got[2] === expected[2]
        @test prepare(evalpoly_cell_reordered)(xs, c) == expected[1]
    end
    xs = collect(range(-2.0, 2.0; length = 300))
    # Not lowered, same values: integer coefficients (the seed's type is not
    # the cell's), a tuple (Base's unrolled method), a two-axis domain.
    @test cell(xs, [1, 2, 3])[1] == evalpoly.(xs, Ref([1, 2, 3]))
    @test cell(xs, Tuple(coefficients))[1] == evalpoly.(xs, Ref(Tuple(coefficients)))
    grid = reshape(xs, 15, 20)
    @test cell(grid, coefficients)[1] == evalpoly.(grid, Ref(coefficients))
    # A one-based view is lowered like a vector.
    @test cell(xs, view(coefficients, 2:6))[1] == evalpoly.(xs, Ref(view(coefficients, 2:6)))
    # Base's errors: empty coefficients have no `c[end]`.
    @test_throws BoundsError cell(xs, Float64[])
    @test cell(Float64[], Float64[]) == (Float64[], 0.0)
    # The output is the only allocation, as for the cell loop.
    allocated(k, xs, c) = @allocated k(xs, c)
    allocated(cell, xs, coefficients)
    allocated(control, xs, coefficients)
    @test allocated(cell, xs, coefficients) <= allocated(control, xs, coefficients) +
                                                _natural_sup_parity_margin
    # The coefficient passes vectorize across cells.
    if Sys.ARCH in (:x86_64, :aarch64)
        @test occursin(r"<\d+ x double>", _plate_entry_llvm(cell, xs, coefficients))
    end
end

# Plates that only feed a lockstep scan run strip by strip with it (snag
# native-lowering-948c4ab6): `_PLATE_STRIP` cells of every such plate, then the
# scan's steps over them, so no plate output is stored at full length. A
# domain that is not a vector (here a tuple) keeps the materialized plates.
# `muladd` leaves contraction to the compiler, so the two lowerings agree to
# rounding.
using ReactiveKernels: scan
@kernel strip_recurrence(steps, pa, pb, q, ea, eb, ca, cb, g0, a0, b0) = begin
    xs = plate(steps) do k
        1 / (k + 0.5)
    end
    sa = plate(xs, Ref(pa)) do x, c
        evalpoly(x, c)
    end
    sb = plate(xs, Ref(pb)) do x, c
        evalpoly(x, c)
    end
    ratio = plate(xs, Ref(q)) do x, c
        evalpoly(x, c)
    end
    out = scan(sa, sb, ratio, Ref(ea), Ref(eb), Ref(ca), Ref(cb);
               init = (g = g0, a = a0, b = b0)) do carry, pa_k, pb_k, q_k, ea, eb, ca, cb
        a = muladd(ea, carry.a, carry.g * pa_k)
        b = muladd(eb, carry.b, carry.g * pb_k)
        ((g = carry.g * q_k, a = a, b = b), ca * a + cb * b)
    end
    return out
end
# The same recurrence with a second, ordinary sequence in lockstep.
@kernel strip_lockstep(steps, weights, pa) = begin
    xs = plate(steps) do k
        1 / (k + 0.5)
    end
    sa = plate(xs, Ref(pa)) do x, c
        evalpoly(x, c)
    end
    out = scan(sa, weights; init = 0.0) do carry, s, w
        next = muladd(w, carry, s)
        (next, next)
    end
    return out
end
# Not strip-fused: an init-including scan, and a plate whose output is also summed.
@kernel strip_include_init(steps, pa) = begin
    sa = plate(steps, Ref(pa)) do k, c
        evalpoly(1 / (k + 0.5), c)
    end
    out = scan(sa; init = 0.0, include_init = true) do carry, s
        (carry + s, carry + s)
    end
    return out
end
@kernel strip_shared_output(steps, pa) = begin
    sa = plate(steps, Ref(pa)) do k, c
        evalpoly(1 / (k + 0.5), c)
    end
    total = sum(sa)
    out = scan(sa; init = 0.0) do carry, s
        (carry + s, carry + s)
    end
    return out, total
end
function _strip_reference(steps, pa, pb, q, ea, eb, ca, cb, g, a, b)
    out = Float64[]
    for k in steps
        x = 1 / (k + 0.5)
        a = muladd(ea, a, g * evalpoly(x, pa))
        b = muladd(eb, b, g * evalpoly(x, pb))
        push!(out, ca * a + cb * b)
        g *= evalpoly(x, q)
    end
    out
end
_strip_fused(kernel) = occursin("_plate_strip_ready", string(code_expr(kernel)))

@testset "authored plate block: plates that only feed a scan run strip by strip" begin
    kernel = prepare(strip_recurrence)
    @test _strip_fused(kernel)
    @test _strip_fused(prepare(strip_lockstep))
    @test !_strip_fused(prepare(strip_include_init))
    @test !_strip_fused(prepare(strip_shared_output))
    pa = [1.0, 0.3, -0.2, 0.05, 0.01]
    pb = [0.8, -0.1, 0.04]
    q = [0.99, 0.002, -0.001]
    coefficients = (pa, pb, q, exp(-0.01), exp(-0.12), 0.3, 0.7, 1.0, 0.0, 0.0)
    for n in (0, 1, 127, 128, 129, 257, 1000)
        steps = 0:n-1
        got = kernel(steps, coefficients...)
        expected = _strip_reference(steps, coefficients...)
        @test length(got) == n
        @test isapprox(got, expected; rtol = 1e-13)
        # The materialized lowering (a tuple domain) agrees.
        # (An empty tuple is type-erased and has no reduction identity.)
        1 <= n <= 300 && @test isapprox(got, kernel(Tuple(steps), coefficients...); rtol = 1e-13)
        # Tuple coefficients take Base's unrolled `evalpoly` in the same strips.
        @test isapprox(kernel(steps, Tuple(pa), Tuple(pb), Tuple(q), coefficients[4:end]...),
                       expected; rtol = 1e-13)
    end
    # A second sequence in lockstep reads the same steps.
    steps, weights = 0:299, collect(range(0.1, 0.9; length = 300))
    lockstep = prepare(strip_lockstep)
    expected = accumulate((c, (s, w)) -> muladd(w, c, s),
                          zip(evalpoly.(1 ./ (steps .+ 0.5), Ref(pa)), weights); init = 0.0)
    @test isapprox(lockstep(steps, weights, pa), expected; rtol = 1e-13)
    @test_throws DimensionMismatch lockstep(steps, weights[1:end-1], pa)
    # Not strip-fused, same values as authored.
    @test prepare(strip_include_init)(0:9, pa) ==
        [0.0; cumsum(evalpoly.(1 ./ ((0:9) .+ 0.5), Ref(pa)))]
    shared = prepare(strip_shared_output)(0:9, pa)
    @test shared[1] == cumsum(evalpoly.(1 ./ ((0:9) .+ 0.5), Ref(pa)))
    # No full-length intermediates: the output plus strip buffers, against the
    # four full-length plate outputs of the materialized lowering.
    steps = 0:6527
    allocated(k, steps) = @allocated k(steps, coefficients...)
    allocated(kernel, steps)
    @test allocated(kernel, steps) < 2 * sizeof(Float64) * length(steps)
end

# A plain scan step's carry-independent statements that read the step's
# elements run as one-cell plates in the scan's strip region (snag
# native-lowering-948c4ab6): the step reads their values from the strip.
# Statements that read the carry, and values that are not concrete numbers,
# stay in the step.
@kernel fission_recurrence(steps, pa, pb, q, ea, eb, ca, cb, g0, a0, b0) = begin
    out = scan(steps, Ref(pa), Ref(pb), Ref(q), Ref(ea), Ref(eb), Ref(ca), Ref(cb);
               init = (g = g0, a = a0, b = b0)) do carry, k, pa, pb, q, ea, eb, ca, cb
        x = 1 / (k + 0.5)
        pa_k = evalpoly(x, pa)
        pb_k = evalpoly(x, pb)
        q_k = evalpoly(x, q)
        a = muladd(ea, carry.a, carry.g * pa_k)
        b = muladd(eb, carry.b, carry.g * pb_k)
        ((g = carry.g * q_k, a = a, b = b), ca * a + cb * b)
    end
    return out
end
# A hoisted value that is a tuple keeps the step as authored.
@kernel fission_tuple_value(steps, w) = begin
    out = scan(steps, Ref(w); init = 0.0) do carry, k, w
        pair = (k * w, k + w)
        (carry + pair[1] * pair[2], carry)
    end
    return out
end
_fission_hoisted(kernel) = occursin("scan_hoisted", string(code_expr(kernel)))

@testset "authored scan: carry-independent step statements run in the strip region" begin
    kernel = prepare(fission_recurrence)
    @test _fission_hoisted(kernel)
    @test _strip_fused(kernel)
    pa = [1.0, 0.3, -0.2, 0.05, 0.01]
    pb = [0.8, -0.1, 0.04]
    q = [0.99, 0.002, -0.001]
    coefficients = (pa, pb, q, exp(-0.01), exp(-0.12), 0.3, 0.7, 1.0, 0.0, 0.0)
    for n in (0, 1, 127, 128, 129, 257, 1000)
        steps = 0:n-1
        expected = _strip_reference(steps, coefficients...)
        got = kernel(steps, coefficients...)
        @test length(got) == n
        @test isapprox(got, expected; rtol = 1e-13)
        @test isapprox(kernel(steps, Tuple(pa), Tuple(pb), Tuple(q), coefficients[4:end]...),
                       expected; rtol = 1e-13)
    end
    # Without a fold, the step stays as authored (cheap carry-independent work
    # already runs beside the carry chain).
    @test !_fission_hoisted(prepare(fission_tuple_value))
    tuple_kernel = prepare(fission_tuple_value)
    @test tuple_kernel(0:9, 0.5) ==
          [sum((k * 0.5) * (k + 0.5) for k in 0:j-1; init = 0.0) for j in 0:9]
    # The output plus strip buffers only.
    steps = 0:6527
    allocated(k, steps) = @allocated k(steps, coefficients...)
    allocated(kernel, steps)
    @test allocated(kernel, steps) < 2 * sizeof(Float64) * length(steps)
end

@testset "tensorized companion: filtered sums and get keep their branches lazy" begin
    # The tensorized rewrite, evaluated on host values, is Base's own fold: the
    # filter is a branch on the accumulator and `get` stays `Base.get`. Over a
    # data-length iterator the fold is one authored loop (retained under a
    # tracing backend); over a helper-produced iterator it is `foldl`.
    native(t, s, u, w, z) = sum(w[j] * u[t - s[j]] for j in eachindex(s) if t > s[j]; init = z)
    looped = ReactiveKernels._kernel_tensorized_rhs(
        :(sum(w[j] * u[t - s[j]] for j in eachindex(s) if t > s[j]; init = z)),
        Set{Symbol}(), nothing, Set([:t, :s, :u, :w, :z]))
    @test looped.head === :let && occursin("@trace", string(looped))
    folded = ReactiveKernels._kernel_tensorized_rhs(
        :(sum(w[j] * u[t - s[j]] for j in _natural_sup_slots(s) if t > s[j]; init = z)))
    @test folded.head === :call && folded.args[1] == GlobalRef(Base, :foldl)
    @test occursin("_recurrence_branch", string(folded))
    u = [1.5, -2.25, 0.125, 7.0, 3.5]
    cases = ((5, [0, 2, 4], [1.0, 0.5, 2.0], 0.0),
             (1, [0, 2, 4], [1.0, 0.5, 2.0], 0.0),     # one dose given
             (1, [1, 2, 4], [1.0, 0.5, 2.0], 0.0),     # none given: init
             (3, Int[], Float64[], 0.0),               # no doses
             (5, [0, 2, 4], [1.0, 0.5, 2.0], 0),       # Int init, Float64 terms
             (5, [0, 2, 4], [1.0, 0.5, 2.0], -0.0))
    for lowered in (looped, folded)
        companion = Core.eval(@__MODULE__, :((t, s, u, w, z) -> $lowered))
        # The closure is defined in a newer world than this running testset;
        # Julia 1.12 and later refuse the direct call (`MethodError`).
        for (t, s, w, z) in cases
            @test Base.invokelatest(companion, t, s, u, w, z) === native(t, s, u, w, z)
        end
    end
    # Only the `init` form is rewritten: without it Base seeds the sum with the
    # first ACCEPTED element, which a traced condition cannot select.
    plain = ReactiveKernels._kernel_tensorized_rhs(:(sum(u[j] for j in r if c[j])))
    @test !occursin("foldl", string(plain))
    # `get` routes through the tensorized hook, whose fallback is `Base.get`.
    @test occursin("_tensorized_get", string(ReactiveKernels._kernel_tensorized_rhs(:(get(u, i, 0.0)))))
    @test ReactiveKernels._tensorized_get(u, 9, 0.0) === 0.0
    @test ReactiveKernels._tensorized_get(u, 2, 0.0) === -2.25
    @test ReactiveKernels._tensorized_get(Dict(:a => 1), :b, 0) === 0
end

module TraceableHelperFixtures
using ReactiveKernels
struct Lattice
    shifts::Vector{Int}
end
struct Exact end
@traceable row_index(t, p::Lattice, i) = t - p.shifts[i]
@traceable row_index(row, ::Exact, i) = row[i]
@traceable function clipped(x, lo = 0.0)::Float64
    y = x - lo
    if y > 0
        return y
    else
        return 0.0
    end
end
@traceable scaled(x::T, s) where {T<:Real} = s * x
# A hand-written tracing method: the native method is an opaque in-place loop,
# the traced one a different implementation of the same mathematics.
function superpose(plan::Lattice, units, weights, nobs)
    out = zeros(nobs)
    for (s, w) in zip(plan.shifts, weights), t in (s + 1):nobs
        out[t] += w * units[t - s]
    end
    out
end
plain_helper(plan, units, weights, nobs) = superpose(plan, units, weights, nobs)
ReactiveKernels.traced(::typeof(superpose), plan::Lattice, units, weights, nobs) =
    [sum(weights[j] * get(units, t - plan.shifts[j], 0.0) for j in eachindex(weights); init = 0.0)
     for t in 1:nobs]
end

@testset "@traceable: the method as written, plus the rewritten one for tracing" begin
    H = TraceableHelperFixtures
    lattice = H.Lattice([0, 2, 5])
    # The native method is the authored one.
    @test H.row_index(7, lattice, 2) == 5
    @test H.row_index([4, 9, 1], H.Exact(), 2) == 9
    @test H.clipped(2.5, 1.0) === 1.5 && H.clipped(0.5, 1.0) === 0.0 && H.clipped(2.0) === 2.0
    @test H.scaled(2, 1.5) == 3.0
    # The tensorized hook dispatches to the rewritten body by the authored
    # signature; on host values it computes what the method computes.
    call = ReactiveKernels.traced
    @test call(H.row_index, 7, lattice, 2) == 5
    @test call(H.row_index, [4, 9, 1], H.Exact(), 2) == 9
    @test call(H.clipped, 2.5, 1.0) === 1.5 && call(H.clipped, 0.5, 1.0) === 0.0
    @test call(H.clipped, 2.0) === 2.0
    @test call(H.scaled, 2, 1.5) == 3.0
    @test call(+, 1, 2) == 3
    # A hand-written method is reached the same way and agrees with the native one.
    units = [1.0, 0.5, 0.25, 0.125, 2.0, 4.0]
    @test call(H.superpose, lattice, units, [1.0, 2.0, 3.0], 6) ==
          H.superpose(lattice, units, [1.0, 2.0, 3.0], 6)
    # The rewritten body carries the recipe rewrites (a traced index reads
    # through `_tensorized_getindex`; a branch is lazy).
    rewritten = string(ReactiveKernels._traceable_definition(
        :(row_index(t, p::Lattice, i) = t - p.shifts[i]), H))
    @test occursin("_tensorized_getindex", rewritten)
    @test occursin("_recurrence_branch", string(ReactiveKernels._traceable_definition(
        :(f(x) = x > 0 ? x : zero(x)), H)))
    # A recipe routes calls to helpers through the hook; Base and package
    # calls, and calls with keyword arguments, keep their form.
    lower(ex) = string(ReactiveKernels._kernel_tensorized_rhs(ex, Set{Symbol}(), H,
                                                              Set([:t, :p, :i, :x])))
    @test occursin("ReactiveKernels.traced", lower(:(row_index(t, p, i))))
    @test occursin("ReactiveKernels.traced", lower(:(not_yet_defined(t))))
    @test !occursin("ReactiveKernels.traced", lower(:(exp(x) + sum(x))))
    @test !occursin("ReactiveKernels.traced", lower(:(plate(x))))
    @test !occursin("ReactiveKernels.traced", lower(:(row_index(t, p; i = i))))
    # A whole recipe `x = f(ports...)` keeps `f` as its raw operation unless
    # `f` has a `traced` method, which sends it through the tracing companion.
    operation(ex) = ReactiveKernels._kernel_operation(
        ex, Symbol[ex.args[2:end]...], Set{Symbol}(ex.args[2:end]); mod = H)
    @test operation(:(superpose(plan, u, w, n))) isa Expr
    @test operation(:(row_index(t, p, i))) isa Expr          # @traceable defines one
    @test operation(:(plain_helper(plan, u, w, n))) === :plain_helper
    # Loud refusals.
    define(ex) = ReactiveKernels._traceable_definition(ex, H)
    @test_throws ArgumentError define(:(sum(x::Lattice) = 0))
    @test_throws ArgumentError define(:(g(x; k = 1) = x))
    @test_throws ArgumentError define(:(g(x) = (x > 0 && return 1; 2)))
    @test_throws ArgumentError define(:(g(x)))
    @test_throws ArgumentError define(:((x -> x)(y) = y))
end
