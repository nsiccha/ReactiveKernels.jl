# Run in a fresh process so earlier tests cannot warm the preference caches.
using ReactiveKernels, Test
const RK = ReactiveKernels

mktempdir() do dir
    write(joinpath(dir, "Project.toml"), """
        [deps]
        ReactiveKernels = "$(Base.PkgId(RK).uuid)"

        [preferences.ReactiveKernels]
        simplify = false
        bound_statistics = false
        type_stability = "error"
        type_stability_level = "values"
        cell_locality = "ignore"
        branch_arms = "all"
        ad_allocations = "warn"
        plate_reverse = "on"
        """)
    pushfirst!(LOAD_PATH, dir)
    try
        workers = 1:16
        ready = Channel{Nothing}(length(workers))
        start = Base.Event()
        tasks = map(workers) do i
            Threads.@spawn begin
                put!(ready, nothing)
                wait(start)
                g = Graph()
                x = value!(g, :x, Float64)
                # Half of the preparations inherit overrides, the others use
                # the project. Each task owns its graph.
                scoped = isodd(i) ? true : nothing
                result = RK._with_settings(scoped, scoped) do
                    k = prepare(g; have = (x,), want = (x,))
                    (k(3.0), RK._simplify(), RK._bound_statistics())
                end
                (result, RK._simplify(), RK._bound_statistics(), RK._type_stability(),
                 RK._type_stability_level(), RK._cell_locality_preference(), RK._branch_arms(),
                 RK._ad_allocations(), RK._plate_reverse_enabled())
            end
        end
        foreach(_ -> take!(ready), workers)
        notify(start)
        for (i, result) in enumerate(fetch.(tasks))
            @test result == ((3.0, isodd(i), isodd(i)), false, false, :error,
                             :values, :ignore, :all, :warn, true)
        end
    finally
        popfirst!(LOAD_PATH)
    end
end
