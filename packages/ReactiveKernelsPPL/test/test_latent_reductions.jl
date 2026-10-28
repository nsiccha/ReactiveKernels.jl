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

@testset "literal latent domains are independent of response lengths" begin
    for (n, n1, n2, position) in ((9, 2, 3, :inline),
        (9, 5, 11, :named), (9, 0, 3, :derived), (1, 2, 3, :inline),
        (0, 2, 3, :inline), (9, 2, 3, :indexed))
        fx = _latent_reduction_fixture(:sum, n; iterator=:literal, n1, n2, position)
        @test only(fx.bound.plate_parameters).range == 1:n
        @test coordinate_names(fx.built.layout) == [Symbol("z.$i") for i in 1:n]
        _check_latent_reduction(fx)
    end
    # A literal domain needs no response consumer to establish its size.
    ast = quote
        @plate for i in 1:(4 + 5)
            z[i] ~ Normal(0, 1)
        end
    end
    bound = bind_data(lower_rkppl(ast, ()), NamedTuple())
    built = build_kernel(bound)
    @test built.layout.total == 9
    @test Base.invokelatest(prepare_query(built, bound, :prior), zeros(9)) ≈
        9logpdf(Normal(), 0.0)
end

@testset "literal latent uses retain their likelihood dimensions" begin
    # refused: the authored response loop indexes beyond the declared latent
    # vector (standing @rkppl language principle 3: Julia indexing safety).
    @test_throws "does not cover the authored response range" _latent_reduction_fixture(
        :sum, 9; iterator=:literal, n2=10, position=:indexed)
    for iterator in (:(1:9), :(eachindex(domain)))
        ast = quote
            @plate for i in $iterator
                z[i] ~ Normal(0, 1)
            end
            y .~ Normal.(z, 1)
        end
        data = (; domain=collect(1:9), y=zeros(9))
        inputs = iterator == :(1:9) ? (; data.y) : data
        bound = bind_data(lower_rkppl(ast, inputs; conditioned=(:y,)), inputs)
        @test build_kernel(bound).layout.total == 9
        bad = merge(inputs, (; y=zeros(3)))
        # refused: broadcasting lengths 9 and 3 is invalid Julia semantics
        # (standing @rkppl language principle 3).
        @test_throws "cannot broadcast" bind_data(
            lower_rkppl(ast, bad; conditioned=(:y,)), bad)
    end
end
