using ReactiveKernels
using ReactiveKernelsPPLExamples
using Test

# The GP modules are loaded, but no GP demo tail or prepared kernel has run.
@test ReactiveKernelsPPLExamples._DEMO_TAIL_EXECUTIONS[] == 0

function _first_use_gp_regr()
    graph = ReactiveKernelsPPLExamples.GPRegrExample.build_gp_regr_graph()
    kernel = prepare(graph;
        have = (:unconstrained, :x, :y), want = :posterior,
        bound = (; x = ReactiveKernelsPPLExamples.GPRegrExample.GP_REGR_X,
                   y = ReactiveKernelsPPLExamples.GPRegrExample.GP_REGR_Y))
    kernel([1.0, 0.5, -0.5])
end

function _first_use_accel_gp()
    module_ = ReactiveKernelsPPLExamples.AccelGPExample
    graph = module_.build_accel_gp_graph()
    kernel = prepare(graph;
        have = (:unconstrained, :Y, :Xgp_1, :slambda_1, :Xgp_sigma_1, :slambda_sigma_1),
        want = :posterior,
        bound = (; Y = module_.ACCEL_GP_Y, Xgp_1 = module_.ACCEL_GP_XGP,
                   slambda_1 = module_.ACCEL_GP_SLAMBDA,
                   Xgp_sigma_1 = module_.ACCEL_GP_XGP_SIGMA,
                   slambda_sigma_1 = module_.ACCEL_GP_SLAMBDA_SIGMA))
    q = vcat([-13.0, 0.5, -1.9], zeros(size(module_.ACCEL_GP_XGP, 2)),
             [0.0, 0.0, -1.9], zeros(size(module_.ACCEL_GP_XGP_SIGMA, 2)))
    kernel(q)
end

function _first_use_gp_pois_regr()
    module_ = ReactiveKernelsPPLExamples.GPPoisRegrExample
    graph = module_.build_gp_pois_regr_graph()
    kernel = prepare(graph;
        have = (:unconstrained, :x, :k), want = :posterior,
        bound = (; x = module_.GP_POIS_X, k = module_.GP_POIS_K))
    kernel(vcat([1.5, 0.0], zeros(11)))
end

function _first_use_hierarchical_gp()
    module_ = ReactiveKernelsPPLExamples.HierarchicalGPExample
    graph = module_.build_hierarchical_gp_graph()
    kernel = prepare(graph;
        have = (:unconstrained, :y, :year_ind, :state_ind, :region_ind,
                :state_region_ind, :N_years, :N_regions, :N_states, :N_years_obs),
        want = :posterior,
        bound = (; y = module_.HGP_Y, year_ind = module_.HGP_YEAR_IND,
                   state_ind = module_.HGP_STATE_IND,
                   region_ind = module_.HGP_REGION_IND,
                   state_region_ind = module_.HGP_STATE_REGION_IND,
                   N_years = module_.HGP_N_YEARS, N_regions = module_.HGP_N_REGIONS,
                   N_states = module_.HGP_N_STATES,
                   N_years_obs = module_.HGP_N_YEARS_OBS))
    q = zeros(933)
    q[end] = 0.5
    kernel(q)
end

@testset "GP first public use in one ordinary function" begin
    @testset "gp_regr" begin
        value = _first_use_gp_regr()
        @test isfinite(value)
        @test _first_use_gp_regr() == value
    end
    @testset "accel_gp" begin
        value = _first_use_accel_gp()
        @test isfinite(value)
        @test _first_use_accel_gp() == value
    end
    @testset "gp_pois_regr" begin
        value = _first_use_gp_pois_regr()
        @test isfinite(value)
        @test _first_use_gp_pois_regr() == value
    end
    @testset "hierarchical_gp" begin
        value = _first_use_hierarchical_gp()
        @test isfinite(value)
        @test _first_use_hierarchical_gp() == value
    end

    @test ReactiveKernelsPPLExamples._DEMO_TAIL_EXECUTIONS[] == 0
end
