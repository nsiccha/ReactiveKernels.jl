# Focused artifact tests for the five-program three-pane docs renderer. Runs in the
# docs environment (Documenter + DI + Enzyme):
#
#   julia --project=docs docs/test_artifacts.jl
#
# Proves the one-to-one coverage gate is REAL (passes on the live five programs,
# fails on a tampered inventory) and that every panel renders actual generated code.

using Test
using Markdown
using ReactiveKernels
include(joinpath(@__DIR__, "kernel_examples.jl"))
const RKD = ReactiveKernelsDocs

module _ArtifactSandbox end

@testset "five-program docs artifacts" begin
    md = RKD.render_five_programs(_ArtifactSandbox)     # build-executes + asserts coverage
    @test md isa Markdown.MD

    # Rebuild the artifacts directly to inspect the coverage contract.
    arts = Any[]
    for snippet in RKD._FIVE_PROGRAM_SNIPPETS
        RKD._evaluate_source(_ArtifactSandbox, snippet.source)
        object = Core.eval(_ArtifactSandbox, :object)
        push!(arts, RKD.program_artifact(snippet.name, snippet.origin, snippet.source,
                                         object, snippet.getter; note = snippet.note))
    end

    @testset "exactly the five programs, each a live ReactiveProgram" begin
        @test Set(a.name for a in arts) == Set(keys(RKD._FIVE_PROGRAM_INVENTORY))
        @test length(arts) == 5
        for a in arts
            @test a.program isa ReactiveKernels.ReactiveProgram
            @test a.program === reactive_program(a.object)
            @test a.dag === a.program.plan                       # DAG identity
            handle = RKD._artifact_handles(a.object)[a.getter]
            @test a.generated == code_expr(a.program, handle)    # generated identity
            @test a.generated isa Expr                           # REAL generated code
        end
    end

    @testset "coverage gate passes on the live five" begin
        @test RKD.assert_program_coverage(arts, RKD._FIVE_PROGRAM_INVENTORY) === arts
    end

    @testset "NUTS group inventory covers the required reactive surface" begin
        nuts = only(a for a in arts if a.name === :nuts_group)
        req = (:chol_metric,                                     # cholesky/velocity
               :init_valgrad, :fwd_valgrad, :bwd_valgrad,        # owned VG bundles
               :init_kinetic, :fwd_kinetic, :bwd_kinetic,        # owned kinetic bundles
               :init_dpot_dpos, :fwd_dpot_dpos, :bwd_dpot_dpos,  # gradient projections
               :init_dham_dmom, :fwd_dham_dmom, :bwd_dham_dmom,  # velocity projections
               :init_ham, :fwd_ham, :bwd_ham,                    # per-endpoint hamiltonians
               :active_ham, :dham, :diverged)                    # selection + energy error
        @test all(r -> r in nuts.inventory.derived, req)
        # control + snapshot surface are HAVE sources (not derived).
        ctrl = (:gofwd, :may_sample, :may_continue, :depth, :n_steps, :acceptance_sum,
                :last_energy_error, :last_diverged, :min_dham)
        @test all(c -> c in nuts.inventory.sources, ctrl)
        @test nuts.inventory.recipe_count == 25
        # the shown dham getter is a DERIVED fused getter (not a source read).
        @test !nuts.getter_is_source
    end

    @testset "state-only programs render a real source-slot getter" begin
        for name in (:welford_variance, :trajectory_stats, :sampling_stats)
            a = only(x for x in arts if x.name === name)
            @test isempty(a.inventory.derived)                   # no reactive DAG nodes
            @test a.getter_is_source                             # pane shows a source getter
            @test a.generated isa Expr                           # still REAL generated code
        end
    end

    @testset "coverage gate is NON-VACUOUS (fails on tampering)" begin
        # A missing program is rejected.
        @test_throws ErrorException RKD.assert_program_coverage(arts[1:4], RKD._FIVE_PROGRAM_INVENTORY)
        # A diverged inventory (drop one required derived getter) is rejected.
        tampered = Dict(RKD._FIVE_PROGRAM_INVENTORY)
        base = RKD._FIVE_PROGRAM_INVENTORY[:nuts_group]
        tampered[:nuts_group] = (derived = Base.setdiff(base.derived, (:dham,)),
                                 sources = base.sources)
        @test_throws ErrorException RKD.assert_program_coverage(arts, tampered)
    end
end
