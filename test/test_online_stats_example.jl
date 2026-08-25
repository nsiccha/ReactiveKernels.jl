using ReactiveKernels
using Statistics
using Test

include(joinpath(@__DIR__, "..", "examples", "online_stats.jl"))
const OSE = OnlineStatsExample

@testset "OnlineStats-style immutable accumulators" begin
    @test isbitstype(OSE.MomentsAccumulator{Float64})
    @test_throws MethodError OSE.MomentsAccumulator(Int)
    @test_throws ArgumentError OSE.MomentsAccumulator{Float64}(-1, 0, 0)
    @test_throws ArgumentError OSE.MomentsAccumulator{Float64}(2, 0, -1)
    @test_throws ArgumentError OSE.MomentsAccumulator{Float64}(2, 0, -Inf)
    @test_throws DomainError OSE._nonnegative_m2(-Inf, Inf)
    @test isnan(OSE._nonnegative_m2(NaN, 1.0))
    @test OSE._nonnegative_m2(Inf, Inf) == Inf
    @test isnan(OSE.MomentsAccumulator{Float64}(2, 0, NaN).m2)
    @test OSE.MomentsAccumulator{Float64}(2, 0, Inf).m2 == Inf

    @testset "empty, singleton, and non-finite semantics" begin
        empty_state = OSE.MomentsAccumulator()
        singleton = @inferred OSE.update(empty_state, 3.0)
        empty32 = OSE.MomentsAccumulator(Float32)
        singleton32 = @inferred OSE.update(empty32, 3)

        @test isnan(mean(empty_state))
        @test isnan(var(empty_state))
        @test isnan(var(empty_state; corrected=false))
        @test mean(singleton) === 3.0
        @test isnan(var(singleton))
        @test var(singleton; corrected=false) === 0.0
        @test mean(empty32) isa Float32
        @test var(empty32) isa Float32
        @test mean(singleton32) === 3.0f0
        @test var(singleton32; corrected=false) === 0.0f0

        nan_state = OSE.update(empty_state, NaN)
        inf_state = OSE.update(empty_state, Inf)
        @test isnan(mean(nan_state))
        @test isnan(var(nan_state))
        @test mean(inf_state) == Inf
        @test isnan(var(inf_state))
        @test isnan(mean(OSE.update(inf_state, 1.0)))
    end

    @testset "streaming updates match stable batch references" begin
        data = Float64[3, -1, 4, 1, 5, -9, 2, 6, 5, 3, 5]
        state = OSE.fit(data)
        @test state.n == length(data)
        @test mean(state) ≈ mean(data)
        @test var(state) ≈ var(data)
        @test var(state; corrected=false) ≈ var(data; corrected=false)
        @test state.m2 >= 0

        constant_data = fill(1.0e12, 1_001)
        constant_state = OSE.fit(constant_data)
        @test mean(constant_state) == 1.0e12
        @test var(constant_state) == 0.0
        @test constant_state.m2 == 0.0

        offset_data = 1.0e12 .+ collect(1.0:1_000.0)
        offset_state = OSE.fit(offset_data)
        @test mean(offset_state) ≈ mean(offset_data)
        @test var(offset_state) ≈ var(offset_data) rtol=1e-12
        @test offset_state.m2 >= 0
    end

    @testset "partition merges are exact identities and associative numerically" begin
        data = sin.(collect(1.0:103.0)) .+ 1.0e6
        ranges = (1:1, 2:4, 5:21, 22:103)
        parts = map(range -> OSE.fit(@view data[range]), ranges)
        empty_state = OSE.MomentsAccumulator()

        @test isequal(merge(empty_state, parts[1]), parts[1])
        @test isequal(merge(parts[1], empty_state), parts[1])

        merged = reduce(merge, parts)
        streamed = OSE.fit(data)
        @test merged.n == streamed.n
        @test mean(merged) ≈ mean(streamed) rtol=1e-14
        @test var(merged) ≈ var(streamed) rtol=1e-10
        @test merged.m2 >= 0

        left_grouped = merge(merge(parts[1], parts[2]),
                             merge(parts[3], parts[4]))
        right_grouped = merge(parts[1],
                              merge(parts[2], merge(parts[3], parts[4])))
        @test mean(left_grouped) ≈ mean(right_grouped) rtol=1e-14
        @test left_grouped.m2 ≈ right_grouped.m2 rtol=1e-10

        # Float16 cannot represent 80_000 as a finite value. Count ratios and
        # the Chan cross term must therefore be evaluated in widened arithmetic
        # before the resulting sufficient statistics are stored back in Float16.
        zeros16 = OSE.fit(Float16, zeros(Float16, 40_000))
        ones16 = OSE.fit(Float16, ones(Float16, 40_000))
        forward16 = @inferred merge(zeros16, ones16)
        reverse16 = @inferred merge(ones16, zeros16)
        for combined in (forward16, reverse16)
            @test combined.n == 80_000
            @test mean(combined) isa Float16
            @test isfinite(mean(combined))
            @test mean(combined) ≈ Float16(0.5)
            @test isfinite(combined.m2)
            @test combined.m2 ≈ Float16(20_000)
            @test @inferred(var(combined)) isa Float16
            @test isfinite(var(combined))
            @test var(combined) ≈ Float16(0.25)
            @test var(combined; corrected=false) ≈ Float16(0.25)
        end
        @test isequal(forward16, reverse16)
        advanced16 = @inferred OSE.update(forward16, one(Float16))
        @test advanced16.n == 80_001
        @test isfinite(advanced16.mean)
        @test isfinite(advanced16.m2)
        @test isfinite(var(advanced16))

        uneven16 = (
            OSE.fit(Float16, zeros(Float16, 30_000)),
            OSE.fit(Float16, ones(Float16, 40_000)),
            OSE.fit(Float16, fill(Float16(2), 10_000)),
        )
        left16 = merge(merge(uneven16[1], uneven16[2]), uneven16[3])
        right16 = merge(uneven16[1], merge(uneven16[2], uneven16[3]))
        @test left16.n == right16.n == 80_000
        @test isfinite(left16.mean) && isfinite(right16.mean)
        @test left16.mean ≈ Float16(0.75) atol=eps(Float16)
        @test right16.mean ≈ Float16(0.75) atol=eps(Float16)
        @test isfinite(left16.m2) && isfinite(right16.m2)
        @test left16.m2 ≈ right16.m2 rtol=Float16(4) * eps(Float16)
        @test isfinite(var(left16)) && isfinite(var(right16))
    end

    @testset "generated state kernels are inferred and allocation-free" begin
        model = OSE.build_online_stats_graph()
        @test model isa KernelSpec
        @test inputs(model) == (model.state, model.observation)
        @test outputs(model) ==
              (model.updated, model.average, model.sample_variance)

        update_kernel = prepare(model;
            have=(:state, :observation), want=:updated)
        merge_kernel = prepare(model;
            have=(:left_partition, :right_partition), want=:merged)
        summary_kernel = prepare(model;
            have=(:state, :observation), want=(:average, :sample_variance))

        empty_state = OSE.MomentsAccumulator()
        one_state = @inferred update_kernel(empty_state, 1.0)
        two_state = @inferred update_kernel(one_state, 3.0)
        @test @inferred(merge_kernel(one_state, two_state)) isa
              OSE.MomentsAccumulator{Float64}
        @test @inferred(summary_kernel(one_state, 3.0)) == (2.0, 2.0)

        report = OSE.kernel_performance_report(update_kernel; iterations=10_000)
        @test report.allocated_bytes == 0
        @test report.elapsed_ns > 0
        @test report.nanoseconds_per_update > 0
        @test report.result.n == 10_000

        reactive_report = OSE.reactive_performance_report(model; iterations=20)
        @test reactive_report.allocated_bytes > 0
        @test reactive_report.elapsed_ns > 0
        @test reactive_report.result.n == 20
    end

    @testset "ReactiveState replacement, invalidation, and frozen cut points" begin
        model = OSE.build_online_stats_graph()
        state = ReactiveState(model; materialize=(model.updated,))
        empty_state = OSE.MomentsAccumulator()

        set!(state, model.state, empty_state)
        set!(state, model.observation, 1.0)
        first_state = get!(state, model.updated)
        @test first_state.n == 1

        # Replacing only the observation invalidates the materialized update and
        # recomputes from the same authoritative source state; it is not an
        # implicit mutable fold.
        set!(state, model.observation, 5.0)
        replacement = get!(state, model.updated)
        @test replacement.n == 1
        @test mean(replacement) == 5.0

        # A caller advances the fold explicitly by promoting the returned state
        # to the next authoritative source value.
        set!(state, model.state, first_state)
        advanced = get!(state, model.updated)
        @test advanced.n == 2
        @test mean(advanced) == 3.0

        freeze!(state, model.updated, advanced)
        set!(state, model.state, empty_state)
        set!(state, model.observation, 99.0)
        @test get!(state, model.average) == 3.0
        unfreeze!(state, model.updated)
        @test get!(state, model.average) == 99.0
    end

    @testset "canonical colored DAG exposes update structure" begin
        model = OSE.build_online_stats_graph()
        update_plan = plan(model;
            have=(:state, :observation),
            want=(:updated, :average, :sample_variance))
        view = visualize(update_plan)
        html = sprint(show, MIME"text/html"(), view)
        dot = dot_source(view)

        @test occursin("class=\"rk-node value have\"", html)
        @test occursin("class=\"rk-node value want\"", html)
        @test occursin("rk-node recipe selected", html)
        @test occursin("#dcfce7", dot)
        @test occursin("#ffedd5", dot)
        @test occursin("#dbeafe", dot)
        @test occursin("update", dot)
    end

    @testset "public docs keep a literal declarative source block" begin
        page = read(joinpath(@__DIR__, "..", "docs", "src", "online-stats.md"),
                    String)
        @test occursin("updates = @kernel begin", page)
        @test occursin("partitions = @kernel begin", page)
        @test occursin("model = merge(updates, partitions)", page)
        @test occursin("Main.ReactiveKernelsDocs.execute_example", page)
        @test !occursin("include(", page)
        @test !occursin("Graph()", page)
        @test !occursin("value!(", page)
        @test !occursin("add!(", page)
    end
end
