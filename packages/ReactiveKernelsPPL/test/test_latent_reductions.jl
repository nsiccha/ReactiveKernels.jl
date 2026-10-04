include("fixtures/latent_reductions.jl")

@testset "whole latent plate reductions retain their graph dependencies" begin
    for fn in (:sum, :mean, :std, :var, :minimum, :maximum, :length), n in (3, 9)
        _check_latent_reduction(_latent_reduction_fixture(fn, n))
    end
    for position in (:named, :derived), n in (3, 9)
        _check_latent_reduction(_latent_reduction_fixture(:sum, n; position))
    end
    for fn in (:sum, :length), n in (0, 1)
        _check_latent_reduction(_latent_reduction_fixture(fn, n))
    end
end
