module PathfinderExternalCorpus

include("reactivehmc_algorithm_corpus.jl")
using .ReactiveHMCAlgorithmCorpus: ENTRY_FIELDS

export CORPUS, ENTRY_FIELDS, PAPER_AUTHORITY

const PAPER_AUTHORITY = (
    repository = "https://jmlr.org/papers/v23/21-0889.html",
    revision = "JMLR 23(306):1-49, 2022 / manuscript 21-0889",
    source = "https://jmlr.org/papers/volume23/21-0889/21-0889.pdf",
    source_sha256 = "8fe38816d4953e5b4e01a8b531abb9f3ea1d1f92041f6c2a7ce5e9c7037c8435",
    algorithms = (1, 3, 4),
)

# Pathfinder is external to the exhaustive ReactiveHMC corpus.  It uses the
# parent's common row schema while remaining a separate collection.
const CORPUS = (
    (
        id = :single_path_pathfinder,
        family = :quasi_newton_variational_inference,
        upstream = PAPER_AUTHORITY,
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
)

end # module PathfinderExternalCorpus
