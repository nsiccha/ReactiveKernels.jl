# Precompile workload: cache this package's lowering, binding, generation,
# build and query preparation code in the package image, so a fresh process
# does not JIT-compile it on first use. Building a model compiles little
# per-model code; without a workload a fresh process still compiled this
# package code once, and Julia serializes that compilation across threads.
# The workload runs a small synthetic hierarchical regression end to end.
#
# `ext/ReactiveKernelsPPLEnzymeExt.jl` runs the same workload again while its
# own image is generated with Enzyme loaded: loading Enzyme invalidates part of
# the code this image caches, and that image caches it again.
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
    build_kernel(surface)
    return nothing
end

@setup_workload begin
    @compile_workload begin
        _construction_workload()
    end
end
