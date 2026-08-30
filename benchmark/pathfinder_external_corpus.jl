module PathfinderExternalCorpus

include("reactivehmc_algorithm_corpus.jl")
using .ReactiveHMCAlgorithmCorpus: ENTRY_FIELDS

export CORPUS, ENTRY_FIELDS, PAPER_AUTHORITY, PATHFINDER_JL_AUTHORITY

const PAPER_AUTHORITY = (
    repository = "https://jmlr.org/papers/v23/21-0889.html",
    revision = "JMLR 23(306):1-49, 2022 / manuscript 21-0889",
    source = "https://jmlr.org/papers/volume23/21-0889/21-0889.pdf",
    source_sha256 = "8fe38816d4953e5b4e01a8b531abb9f3ea1d1f92041f6c2a7ce5e9c7037c8435",
    algorithms = (1, 3, 4),
)

const PATHFINDER_JL_AUTHORITY = (
    repository = "https://github.com/mlcolab/Pathfinder.jl",
    revision = "dba8c9acc25f2905078d428ddd50b5d9276c3847",
    version = "0.10.7",
    source_sha256 = (
        "src/inverse_hessian.jl" =>
            "7a32c8e5b8359d2c7d813cae21885e7cafcb357cc1d15e8aebd5f020f38d3309",
        "src/mvnormal.jl" =>
            "9083a7856ddf4b2bf3199bff62487dddfbdfd7a5de54264c8ac04b26ef2e29b8",
        "src/elbo.jl" =>
            "5a35f03afb93fd6657f6a86040b1a4010e719022e3d9e9bf412a7303419e2611",
        "src/singlepath.jl" =>
            "972b3a1d206887c38ec9cd69d22c664f8103ad2f9657021f860cc485c0aba187",
    ),
)

const UPSTREAM_AUTHORITY = (
    paper = PAPER_AUTHORITY,
    implementation = PATHFINDER_JL_AUTHORITY,
)

# Pathfinder is external to the exhaustive ReactiveHMC corpus.  It uses the
# parent's common row schema while remaining a separate collection.
const CORPUS = (
    (
        id = :single_path_pathfinder,
        family = :quasi_newton_variational_inference,
        upstream = UPSTREAM_AUTHORITY,
        current_reactive_sources = (
            "benchmark/pathfinder_kernel_authoring_fixture.jl",
        ),
        capabilities = (
            :captured_callable_and_constant_ports,
            :linear_algebra,
            :factorization,
            :data_dependent_select,
            :batched_draw_transform,
            :reduction,
            :explicit_rng_effects,
        ),
        oracle = (
            execution = :separate_python_process,
            source = "benchmark/pathfinder_oracle/pathfinder_oracle.py",
            receipt = "benchmark/pathfinder_oracle/pathfinder_oracle.toml",
        ),
        comparison = (
            exact = (:curvature_accepted, :best_index),
            numeric = (
                :alpha_next,
                :covariance,
                :mean,
                :elbo_draws,
                :log_q,
                :elbo,
                :output_draws,
            ),
            atol = 1e-12,
            rtol = 1e-12,
        ),
        minimum_acceptance = :native_and_reactant,
    ),
    (
        id = :pathfinder_jl_compact_lbfgs,
        family = :quasi_newton_variational_inference,
        upstream = UPSTREAM_AUTHORITY,
        current_reactive_sources = (
            "benchmark/pathfinder_jl_kernel_authoring_fixture.jl",
        ),
        capabilities = (
            :captured_callable_and_constant_ports,
            :fixed_capacity_history,
            :compact_lbfgs,
            :triangular_solve,
            :linear_algebra,
            :factorization,
            :batched_draw_transform,
            :reduction,
            :explicit_rng_effects,
        ),
        oracle = (
            execution = :separate_python_process,
            source = "benchmark/pathfinder_oracle/pathfinder_oracle.py",
            receipt = "benchmark/pathfinder_oracle/pathfinder_oracle.toml",
            section = "pathfinder_jl_compact",
        ),
        comparison = (
            exact = (),
            numeric = (
                :history_cross,
                :compact_middle,
                :covariance,
                :mean,
                :elbo_draws,
                :log_q,
                :elbo,
                :output_draws,
            ),
            atol = 1e-12,
            rtol = 1e-12,
        ),
        minimum_acceptance = :native_and_reactant,
    ),
)

end # module PathfinderExternalCorpus
