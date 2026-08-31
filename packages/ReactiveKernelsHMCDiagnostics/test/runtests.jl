using ReactiveKernelsHMCDiagnostics
using ReactiveKernelsNUTSExamples: NUTSDiagnostics
using ReactiveKernelsStreamingStats
using ReactiveKernels: prepare
using Test

@testset "mergeable HMC diagnostics" begin
    records = [
        (diagnostics=NUTSDiagnostics(3, 7, 0.9, false, -0.1), stepsize=0.2),
        (diagnostics=NUTSDiagnostics(5, 15, 0.8, true, -Inf), stepsize=missing),
    ]
    combined = fit_diagnostics(records)
    partitioned = merge(fit_diagnostics(records[1:1]),
                        fit_diagnostics(records[2:2]))

    @test sample_count(combined) == 2
    @test combined.n_divergent == 1
    @test max_tree_depth(combined) == 5
    @test divergence_rate(combined) == 0.5
    @test mean_energy_error(combined) == -0.1
    @test mean_stepsize(combined) == 0.2
    @test sample_count(partitioned) == sample_count(combined)
    @test partitioned.n_divergent == combined.n_divergent
    @test max_tree_depth(partitioned) == max_tree_depth(combined)
    @test mean_acceptance_rate(partitioned) ≈ mean_acceptance_rate(combined)

    online = online_diagnostics()
    fit!(online, records)
    @test ReactiveKernelsHMCDiagnostics.snapshot(online) == combined
    reset!(online)
    @test sample_count(online) == 0
end

@testset "acyclic downstream integration" begin
    report = metric_adaptation_report()
    @test report.count == 4.0
    @test report.dimension == 3

    model = build_partition_graph()
    kernel = prepare(model;
        have=(:left_partition, :right_partition), want=:merged)
    @test kernel(fit([1.0]), fit([2.0])).n == 2
end
