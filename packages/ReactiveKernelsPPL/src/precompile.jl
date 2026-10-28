# Precompile workload: cache this package's lowering, binding, generation,
# build and query preparation code in the package image, so a fresh process
# does not JIT-compile it on first use. Building a model compiles little
# per-model code; without a workload a fresh process still compiled this
# package code once, and Julia serializes that compilation across threads.
# The workload runs a small synthetic hierarchical regression end to end.
@setup_workload begin
    workload_body = :(begin
        a ~ Normal(0, 5)
        b ~ Normal(0, 2)
        s ~ Exponential(1)
        tau ~ Exponential(1)
        c[levels(g)] .~ Normal.(0, tau)
        mu = a .+ b .* x .+ c[g]
        y .~ Normal.(mu, s)
    end)
    workload_columns = Dict{Symbol,AbstractVector}(
        :x => [-1.0, 0.0, 1.0, 0.5], :g => [1, 2, 1, 2], :y => [0.5, 1.0, 2.0, 1.5])
    workload_names = (:y, :x, :g)
    @compile_workload begin
        workload_plan = bind_data(lower_rkppl(workload_body, workload_names; mod = @__MODULE__,
                                     conditioned = workload_names), workload_columns)
        workload_built = build_kernel(workload_plan)
        workload_kernel = prepare_query(workload_built, workload_plan, :sampler)
        Base.invokelatest(workload_kernel, zeros(workload_built.layout.total))
    end
end
