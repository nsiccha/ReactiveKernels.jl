using Test
using ReactiveKernels
using ReactiveKernelsDistributionKernels: DistributionKernelSources
import Distributions

@testset "LogNormal walkthrough executes in an isolated module" begin
    # The docs page supplies only ReactiveKernels. Imports in other tests must
    # not hide a missing dependency in the displayed executable source.
    sandbox = Module(gensym(:LogNormalDocs))
    Core.eval(sandbox, :(using ReactiveKernels))
    Base.include_string(sandbox, DistributionKernelSources.LOGNORMAL_SOURCE,
        "lognormal-docs-source.jl")
    artifact = getfield(sandbox, :docs_example)
    @test artifact.output ≈ Distributions.logpdf(
        Distributions.LogNormal(0.2, 0.9), 1.4) rtol=8eps(Float64)

    object = getfield(sandbox, :lognormal)
    density = prepare(object.logpdf;
        have=(:x, :location, :scale), want=:logpdf)
    inverse_cdf = prepare(object.quantile;
        have=(:p, :location, :scale), want=:quantile)
    for (location, scale) in ((0.2, 0.9), (-0.4, 1.2))
        reference = Distributions.LogNormal(location, scale)
        for x in (0.4, 1.4, 3.0)
            @test density(x, location, scale) ≈
                Distributions.logpdf(reference, x) rtol=8eps(Float64)
        end
        for p in (0.1, 0.5, 0.9)
            @test inverse_cdf(p, location, scale) ≈
                Distributions.quantile(reference, p) rtol=8eps(Float64)
        end
    end
end
