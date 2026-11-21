# Included in CellJointFootprintTests to share its laws and ragged fixture.
include(joinpath(pkgdir(ReactiveKernels), "test", "jax_python.jl"))
include("jax_ppl_data.jl")

struct AnchoredLaw{N}
    n::N
end
ReactiveKernelsPPL.sampling_geometry(::Type{AnchoredLaw}, args, shape) =
    ParameterGeometry((args[1],), :($(args[1]) - 1); support = :real,
        constrain = (u, shape, n) -> vcat(0.0, u),
        unconstrain = (v, shape, n) -> v[2:end], logjac = (u, shape, n) -> 0.0)
ReactiveKernelsPPL.sampling_distribution_graph(::Type{AnchoredLaw}, args::Tuple) = block_density

@testset "JAX joint cell geometry rebinds and returns each constrained row" begin
    empty_data = (Int[], Vector{Int}[])
    for (law, data) in ((:JuliaExpLaw, (([3, 2], [[1], Int[]]),
                                       ([4, 0, 2, 1], [[2, 4], Int[], [1], Int[]]))),
                        (:AnchoredLaw, (([3, 1, 4], [[1], Int[], [4]]),)))
        empty = fixture(law, empty_data...)
        src = jax_source(empty.built, empty.bound)
        @test occursin("UNTRANSLATED = {}", src.text)
        for (n, picks) in (empty_data, data...)
            fx = fixture(law, n, picks)
            @test jax_source(fx.built, fx.bound).text == src.text
            if !rk_jax_available()
                @test_skip false
                continue
            end
            args = (fx.u, jax_raw_data(fx.bound, src)...)
            got = rk_jax_evaluate(src, :log_density, args; grad = :joint, prepare = :arguments)
            if isempty(n)
                @test iszero(got.joint_value)
                @test isempty(got.grad)
            else
                sampler = prepare_sampler(fx.built, fx.bound, fx.u;
                    backend = AutoEnzyme(; mode = Enzyme.Reverse))
                expected, grad = sampler_value_and_gradient!(sampler, similar(fx.u), fx.u)
                @test got.joint_value ≈ expected atol = 1e-10
                @test got.grad ≈ grad atol = 1e-10
            end
            rows = rk_jax_evaluate(src, :constrain, args; prepare = :arguments).value
            @test rows["field_lengths"] == n
            native = constrain(fx.built.layout, fx.u).field
            @test all(i -> rows["field"][i][1:n[i]] ≈ native[i], eachindex(n))
        end
    end
    # The graph endpoint reads u[2:end] on ragged rows. Its logical result
    # is one short on each row, even on the row shorter than the padding.
    fx = fixture(:ShortLaw, [3, 2], [[1], Int[]])
    src = jax_source(fx.built, fx.bound)
    if rk_jax_available()
        args = (fx.u, jax_raw_data(fx.bound, src)...)
        @test_throws "wrong shape" rk_jax_evaluate(src, :constrain, args; prepare = :arguments)
    else
        @test_skip false
    end
end
