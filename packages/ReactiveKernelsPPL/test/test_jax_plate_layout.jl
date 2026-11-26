module JAXPlateLayoutTests

using ReactiveKernels, ReactiveKernelsPPL, Test
include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))
include("jax_ppl_data.jl")
include(joinpath(pkgdir(ReactiveKernels), "test", "jax_guarded_scan_fixture.jl"))

@testset "Sampler plate layouts on observed and computed ragged rows" begin
    model = @rkppl begin
        a ~ Normal(0, 1)
        @plate for i in eachindex(y)
            y[i] .~ Normal.(a .* x[i], 1)
        end
    end
    x = [[1.0, -0.5], Float64[], [2.0], [0.25, 1.25, -0.5]]
    y = [[0.2, -0.1], Float64[], [0.3], [-0.1, 0.6, 0.1]]
    bind(xs, ys) = model(; x = xs) | (; y = ys)
    first_plan = bind(x, y)
    built = build_kernel(first_plan)
    u = [0.35]
    for layout in (:padded, :groups, :chains)
        src = jax_source(built, first_plan; plate_layout = layout)
        if layout === :groups
            @test occursin("def prepare_plate_layout(", src.text)
        end
        rk_jax_available() || continue
        # The same source accepts changed lengths, unit order and number of units.
        for (xs, ys) in ((x, y), (vcat(reverse(x), [[1.2, -0.8, 0.7, 0.4]]),
                                  vcat(reverse(y), [[0.1, 0.2, -0.3, 0.5]])))
            plan = bind(xs, ys)
            query = prepare_query(build_kernel(plan), plan, :sampler)
            slope = -only(u) + sum(sum(row .* (obs .- only(u) .* row))
                                   for (row, obs) in zip(xs, ys))
            result = withenv("RK_JAX_PLATE_GROUPS" => (layout === :groups ? "2" : nothing)) do
                rk_jax_evaluate(src, "log_density", (u, jax_raw_data(plan, src)...);
                                grad = :joint, prepare = :arguments)
            end
            @test result.joint_value ≈ query(u) rtol = 1e-11
            @test only(result.grad) ≈ slope rtol = 1e-11
        end
    end
end

@testset "Sampler entry with selectable subject scans" begin
    model = @rkppl begin
        a ~ Normal(0, 1)
        mu = jax_guarded_scan_rows(a, bases, types, orders)
        y ~ Normal(mu, 1)
    end
    plan = model(; bases = [[-1.0, 0.5], [2.0], Float64[]],
                   types = [[3, 2, 3], [3], Int[]],
                   orders = [[1, -1, 1], [1], Int[]]) | (; y = 0.25)
    built = build_kernel(plan)
    src = jax_source(built, plan; plate_layout = :padded)
    @test occursin("SUBJECT_SCAN_SCHEDULES =", src.text)
    @test any(f -> f.name == "select_plate_layout", src.functions)
    if rk_jax_available()
        query = prepare_query(built, plan, :sampler)
        u = [0.35]
        inputs = (u, jax_raw_data(plan, src)...)
        for walk in ("padded", "events")
            result = withenv("RK_JAX_SUBJECT_WALK" => walk) do
                rk_jax_evaluate(src, "log_density", inputs; grad = :joint, prepare = :arguments)
            end
            @test result.joint_value ≈ query(u) rtol = 1e-11
            @test result.grad ≈ rk_fd_gradient(query, u) rtol = 1e-6
        end
    end
end

end # module
