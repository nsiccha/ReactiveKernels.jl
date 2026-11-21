using Test, ReactiveKernels, ReactiveKernelsPPL
include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))

@testset "partitioned JAX sampler and owned atomic events" begin
    model = @rkppl begin
        location ~ Normal(0.3, 1.7)
        z[1:K] .~ LogNormal.(location, 0.9)
        joint ~ Dirichlet(alpha)
        y .~ Normal.(z[group], 0.7)
    end
    raw = (; K = 3, alpha = [1.2, 2.1, 1.5], group = [1, 2, 1], y = [0.4, -0.1, 0.7])
    labels = [10, 21, 10]
    binding(data) = model(; Base.structdiff(data, (; y = nothing))...) | (; data.y)
    plan = binding(raw)
    bound = binding(merge(raw, (; station = labels)))
    original, built = build_kernel(plan), build_kernel(bound)
    catalogue = density_factors(original, plan)
    shared = select_factors(catalogue; site = :location)
    units = Dict(i => select_factors(catalogue; site = :z, entries = j)
        for (j, i) in enumerate([10, 21, 42]))
    units[-5] = select_factors(catalogue; site = :joint)
    units[99] = select_factors(catalogue, [])
    @test length(units[-5]) == 2  # one joint prior and one whole simplex Jacobian
    kwargs = (; by = :station, global_factors = shared, unit_factors = units,
        expected_layout = original.layout)
    out = jax_source(built, bound, :y; kwargs...)
    @test "station" ∉ first.(only(f for f in out.source.functions if f.name == "prepare_data").inputs)
    @test out.partition_data.labels == [-5, 10, 21, 42, 99]
    @test_throws ContractValidationError jax_source(built, bound, :y;
        kwargs..., global_factors = units[10])
    changed = binding(merge(raw, (; y = reverse(raw.y), station = labels)))
    @test_throws ContractValidationError jax_source(build_kernel(changed), changed, :y; kwargs...)

    more = merge(raw, (; K = 19, group = repeat(raw.group, 4), y = repeat(raw.y, 4)))
    more_plan = binding(more)
    more_original = build_kernel(more_plan)
    more_catalogue = density_factors(more_original, more_plan)
    more_bound = binding(merge(more, (; station = repeat(labels, 4))))
    more_source = jax_source(build_kernel(more_bound), more_bound, :y; by = :station,
        global_factors = select_factors(more_catalogue; site = :location),
        unit_factors = Dict(10 => select_factors(more_catalogue; site = :z, entries = 1)),
        expected_layout = more_original.layout).source
    @test more_source.text == out.source.text

    if rk_jax_available()
        obs = prepare_partition(built, bound, :y; by = :station)
        global_query = prepare_query(original, plan, shared)
        local_queries = Dict(k => prepare_query(original, plan, s) for (k, s) in units)
        sampler = prepare_query(original, plan, :sampler)
        function native(f, u)
            h = 1e-5
            g = map(eachindex(u)) do i
                a, b = copy(u), copy(u)
                a[i] += h
                b[i] -= h
                (f(a) - f(b)) / (2h)
            end
            (f(u), g)
        end
        batch_labels = [99, 21, 42, -5, 10, 21]
        points = [collect(range(-0.2, 0.3; length = original.layout.total)) .+ j / 20
            for j in eachindex(batch_labels)]
        expected = [native(u -> local_queries[k](u) + (k in labels ? obs(u, k) : 0.0), p)
            for (p, k) in zip(points, batch_labels)]
        mktempdir() do dir
            write(joinpath(dir, "model.py"), out.source)
            write(joinpath(dir, "inputs.py"), """
                import numpy as np
                raw = $(_rk_py_input(merge(raw, (; station = labels))))
                ownership = $(_rk_py_input(out.partition_data))
                points = np.array($(_rk_py(points)))
                labels = np.array($(_rk_py(batch_labels)))
                expected = $(_rk_py_input(Tuple(expected)))
                global_reference = $(_rk_py_input(native(global_query, first(points))))
                full_reference = $(_rk_py_input(native(sampler, first(points))))
                """)
            cp(joinpath(@__DIR__, "jax_partition.py"), joinpath(dir, "run.py"))
            @test success(pipeline(`$RK_JAX_PYTHON $(joinpath(dir, "run.py"))`;
                stdout = stdout, stderr = stderr))
        end
    else
        @test_skip "set RK_JAX_PYTHON to run partition parity"
    end
end
