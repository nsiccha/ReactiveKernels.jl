using Test, ReactiveKernels, ReactiveKernelsPPL, Enzyme, Distributions
using DifferentiationInterface: AutoEnzyme

# The same ordinary functions serve the standalone backend and PPL probes.
include(joinpath(@__DIR__, "../../../benchmark/repro_enzyme_generator_const_array_capture.jl"))
const _NGC = NativeGeneratorCaptureRepro

@testset "ordinary array reader native reverse boundary" begin
    u = [-0.2, 0.1, 0.3, -0.1]
    x = [-0.5, 0.0, 0.5, 1.0]
    idx = [1, 1, 2, 2]
    y = [-1.0, 0.1, 1.0, 0.4]
    inputs = [_NGC.PublicPrepared((; x, fields = ((; idx),)))]
    snapshot = deepcopy((u, x, idx, y, only(inputs).state))
    @test isequal((u, x, idx, y, only(inputs).state), snapshot)
    expected_value = -8.21400826563738
    expected_gradient = [-0.15, 1.25, 0.025, 0.975]

    for reader in (:ordinary_reader, :typed_reader, :inline_collect_reader, :loop_reader,
                   :prepared_reader)
        call = reader === :prepared_reader ? :($reader(b, inputs)) : :($reader(b, x, idx))
        body = quote
            b[1:2, 1:2] .~ Normal.(0, 1)
            out = $call
            y .~ Normal.(out, 1)
        end
        data = reader === :prepared_reader ? (; inputs, y) : (; x, idx, y)
        bound = bind_data(lower_rkppl(body, keys(data); mod = _NGC,
            conditioned = (:y,)), data)
        built = build_kernel(bound)
        sampler = prepare_sampler(built, bound, u;
            backend = AutoEnzyme(; mode = Enzyme.Reverse))
        @test sampler(u) ≈ expected_value
        for _ in 1:2
            g = zero(u)
            result = try
                sampler_value_and_gradient!(sampler, g, u)
            catch err
                err isa Enzyme.Compiler.EnzymeRuntimeActivityError || rethrow()
                println("UNSUPPORTED PPL ", reader, ": ", nameof(typeof(err)))
                nothing
            end
            if reader in (:ordinary_reader, :prepared_reader)
                # Valid authored Julia shape awaiting the upstream capability;
                # never an admission refusal or a substitute AD configuration.
                @test_broken result !== nothing && result[1] ≈ expected_value &&
                    g ≈ expected_gradient
            else
                @test result[1] ≈ expected_value
                @test g ≈ expected_gradient
            end
            @test isequal((u, x, idx, y, only(inputs).state), snapshot)
        end
    end
end
