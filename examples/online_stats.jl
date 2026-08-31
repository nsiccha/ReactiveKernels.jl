module OnlineStatsExample

using ReactiveKernelsStreamingStats
using ReactiveKernelsHMCDiagnostics

export MomentsAccumulator, update, fit
export OnlineMoments, online_moments, update!, fit!, snapshot, reset!
export welford_var, step!
export HMCDiagnosticsAccumulator, record_transition, fit_diagnostics
export OnlineDiagnostics, online_diagnostics, record!
export sample_count, max_tree_depth, divergence_rate, divergence_percent
export mean_tree_depth, mean_leapfrog_steps, mean_acceptance_rate
export mean_energy_error, mean_stepsize
export metric_adaptation_report
export build_partition_graph
export partition_performance_report, diagnostics_performance_report
export reactive_performance_report
export demo

end # module OnlineStatsExample

if abspath(PROGRAM_FILE) == @__FILE__
    OnlineStatsExample.demo()
end
