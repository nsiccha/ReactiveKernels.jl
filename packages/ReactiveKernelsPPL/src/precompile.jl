# Precompile workload: cache this package's lowering, binding, generation,
# build and query preparation code in the package image, so a fresh process
# does not JIT-compile it on first use. Building a model compiles little
# per-model code; without a workload a fresh process still compiled this
# package code once, and Julia serializes that compilation across threads.
# The workload runs a small synthetic hierarchical regression end to end.
#
# The Distributions, WarmupHMC, Enzyme and joint WarmupHMC/Enzyme extensions
# rerun this workload while their images are built. Each caches construction
# with the methods that its dependencies add.
function _construction_workload()
    body = :(begin
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        s ~ Exponential(1)
        tau ~ Exponential(1)
        c[levels(g)] .~ Normal.(0, tau)
        mu = a .+ b .* x .+ c[g]
        y .~ Normal.(mu, s)
    end)
    columns = Dict{Symbol,AbstractVector}(
        :x => [-1.0, 0.0, 1.0, 0.5], :g => [1, 2, 1, 2], :y => [0.5, 1.0, 2.0, 1.5])
    names = (:y, :x, :g)
    plan = bind_data(lower_rkppl(body, names; mod = @__MODULE__, conditioned = names),
        columns)
    built = build_kernel(plan)
    kernel = prepare_query(built, plan, :sampler)
    Base.invokelatest(kernel, zeros(built.layout.total))
    # The same model through the `@rkppl` surface: called with keyword data,
    # then conditioned with `|`. Binding and lowering take their own entry
    # points there (`condition`, `_lower_model_latest`).
    model = RKPPLModel(body, @__MODULE__)
    surface = model(; x = columns[:x], g = columns[:g]) | (; y = columns[:y])
    return build_kernel(surface), surface
end

function _marginalized_workload()
    model = @rkppl begin
        scale ~ Exponential(1)
        @scan begin
            x[1] = 0.0
            for t in 2:T
                innovation ~ Normal(0, 1)
                step = scale * innovation
                x[t] = x[t - 1] + step
            end
        end
        y .~ Normal.(x, 1)
    end
    plan = marginalize(model(), :x) | (; y = [0.2, -0.1, 0.4])
    prepare_query(build_kernel(plan), plan, :likelihood)
    return nothing
end

function _precompile_lowering_calls()
    # The workload exercises these syntax walkers, but Julia 1.10 keeps
    # some calls as inferred IR only. Model construction also dispatches
    # to them from untyped expression trees in a fresh process.
    symbols = Set{Symbol}
    definitions = Dict{Symbol,Any}
    shapes = Dict{Symbol,Symbol}
    for (f, types) in (
            (_canonical_expr, (Expr, symbols, definitions, shapes, _ShapeEnv, String, Bool)),
            (_mentions_array, (Symbol, StructuralPlan, symbols, Bool)),
            (_value_metadata, (StructuralPlan, Expr, symbols, Bool, Bool, Bool)),
            (VectorAssignmentSpec, (Symbol, Expr, Symbol)),
            (_AbsentArray{Float64,1}, (Symbol, Tuple{Int})),
            (_obs_axis, (Expr, symbols, definitions, shapes, symbols, _ShapeEnv)),
            (_program_value_names!, (symbols, Symbol)),
            (_resolve_module_calls, (Symbol, Module, symbols, String)),
            (_shape_of, (Symbol, symbols, definitions, shapes, symbols)),
            (_symbols_in, (Symbol,)),
            (_value_symbols, (Symbol,)),
            (_whole_name_reads!, (symbols, Symbol)))
        precompile(f, types) || error("no $f method for $types")
    end
    # Take the lowering context's concrete NamedTuple type from the
    # construction above, without duplicating its field list here.
    for (f, types) in ((_composed_root, (Symbol, NamedTuple)),
                       (_reads_array_value, (Symbol, NamedTuple, symbols)))
        for instance in Base.specializations(which(f, types))
            sig = instance.specTypes
            isconcretetype(sig) && sig <: Tuple{typeof(f),types...} || continue
            precompile(sig) || error("no lowering method for $sig")
        end
    end
end

@setup_workload begin
    @compile_workload begin
        built, plan = _construction_workload()
        prepare_coordinate_conditional(built, plan, zeros(built.layout.total))
        _marginalized_workload()
        _precompile_lowering_calls()
    end
end
