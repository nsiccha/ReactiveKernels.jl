using ReactiveKernelsPPLExamples.BonesModelExample
using ReactiveKernelsDistributionKernels.DistributionKernelSources: normal
using LogExpFunctions: logistic

# Graph-independent oracle for the BUGS bones graded-response model, computed with
# explicit per-cell branches (no clamp/mask tricks) directly from the raw block.
function _bones_reference(theta, GRADE, GAMMA, DELTA, NCAT)
    nChild, nInd = size(GRADE)
    prior = sum(-0.5 * log(2π) - log(36.0) - 0.5 * (t / 36.0)^2 for t in theta)
    ll = 0.0
    for i in 1:nChild, j in 1:nInd
        g = GRADE[i, j]
        g == -1 && continue
        q_lo = g > 1 ? logistic(DELTA[j] * (theta[i] - GAMMA[j, g - 1])) : 1.0
        q_hi = g < NCAT[j] ? logistic(DELTA[j] * (theta[i] - GAMMA[j, g])) : 0.0
        ll += log(q_lo - q_hi)
    end
    (; prior, log_jacobian = 0.0, likelihood = ll, posterior = prior + ll)
end

_bones_bound() = (; GRADE = BONES_GRADE, GAMMA = BONES_GAMMA,
                    DELTA = BONES_DELTA, NCAT = BONES_NCAT)

@testset "PPL graph — bones (posteriordb)" begin
    artifact = evaluate_bones_model_source()
    @test artifact.source == strip(BONES_SOURCE, '\n')
    @test artifact.output ==
          Base.invokelatest(artifact.kernel, Tuple(artifact.inputs)...)
    model = artifact.model
    nChild = size(BONES_GRADE, 1)
    theta = 0.4 .* sin.(1:nChild)
    reference = _bones_reference(theta, BONES_GRADE, BONES_GAMMA, BONES_DELTA, BONES_NCAT)

    @testset "authored in-graph over ONLY the raw data block" begin
        # Grid coordinates built in-graph from the raw dimensions, not passed in.
        @test occursin("repeat(1:n_child, 1, n_ind)", BONES_SOURCE)
        @test occursin("repeat(permutedims(1:n_ind), n_child, 1)", BONES_SOURCE)
        @test occursin("GAMMA[lo_lin]", BONES_SOURCE)          # raw gamma gathered in-graph
        @test occursin("theta[ROWIDX]", BONES_SOURCE)
        @test occursin("DELTA[COLIDX]", BONES_SOURCE)
        @test occursin("GRADE .!= -1", BONES_SOURCE)           # missing mask in-graph
        @test occursin("log.(arg)", BONES_SOURCE)
        @test occursin("normal(0.0, 36.0).logpdf(t)", BONES_SOURCE)
        @test !occursin("struct ", BONES_SOURCE)
        @test artifact.normal_object === normal
    end

    @testset "acceptance entry is ONLY the rebuildable raw block" begin
        d = ReactiveKernelsPPLExamples._posteriordb_data("bones_data-bones_model")
        rebuilt = bones_inputs(d)
        @test propertynames(rebuilt) == (:GRADE, :GAMMA, :DELTA, :NCAT)   # no coords
        @test rebuilt.GRADE == BONES_GRADE
        @test rebuilt.GAMMA == BONES_GAMMA
        @test rebuilt.DELTA == BONES_DELTA
        @test rebuilt.NCAT == BONES_NCAT
        @test count(!=(-1), BONES_GRADE) == 422    # observed cells in the real data
    end

    @testset "posterior decomposition vs the independent reference oracle" begin
        p = plan(model.graph;
                 have = (model.unconstrained, model.GRADE, model.GAMMA, model.DELTA,
                         model.NCAT),
                 want = (model.prior, model.log_jacobian, model.likelihood, model.posterior))
        prior, log_jacobian, likelihood, posterior =
            prepare(p)(theta, BONES_GRADE, BONES_GAMMA, BONES_DELTA, BONES_NCAT)
        @test prior ≈ reference.prior
        @test log_jacobian ≈ reference.log_jacobian
        @test likelihood ≈ reference.likelihood
        @test posterior ≈ reference.posterior
    end

    @testset "build+prepare+execute in one ordinary function, repeat-use stable" begin
        function once(theta)
            k = prepare(build_bones_model_graph();
                have = (:unconstrained, :GRADE, :GAMMA, :DELTA, :NCAT),
                want = :posterior, bound = _bones_bound())
            k(theta)
        end
        @test once(theta) ≈ reference.posterior
        k = prepare(build_bones_model_graph();
            have = (:unconstrained, :GRADE, :GAMMA, :DELTA, :NCAT),
            want = :posterior, bound = _bones_bound())
        @test k(theta) == k(theta)
    end

    @testset "data-generic: alternate small dataset with ragged ncat" begin
        d = Dict("grade" => [[1, 3], [2, 1]],
                 "gamma" => [[0.5, -1.0], [0.3, 0.9]],
                 "delta" => [1.2, 0.8], "ncat" => [2, 3])
        inp = bones_inputs(d)
        thetas = [0.3, -0.4]
        k = prepare(build_bones_model_graph();
            have = (:unconstrained, :GRADE, :GAMMA, :DELTA, :NCAT),
            want = :posterior, bound = inp)
        @test k(thetas) ≈
              _bones_reference(thetas, inp.GRADE, inp.GAMMA, inp.DELTA, inp.NCAT).posterior
    end
end
