using Test, ReactiveKernels, Random
include("../benchmark/sampler_transpiler/prepared_examples.jl")
const PTE = PreparedTranspilerExamples

# Run this same consumer contract with backend=:reactant after loading Reactant
# in the optional benchmark environment. Each backend has its own source oracle.
function check_transpiled_program(backend)
    @testset "prepared transpiler ($backend)" begin
        result = PTE.scalar_example(; backend)
        @test Float64(result.first.outputs.value) == 9.0
        @test Float64(result.continued.outputs.value) == 21.0
        @test Float64(result.replayed.outputs.value) == 9.0
        vector = PTE.vector_example(; backend)
        @test vector.input == [2.0, 4.0]
        @test Array(vector.first.outputs.squared) == [0.25, 1.0]
        @test Array(vector.continued.outputs.position) == [0.125, 0.25]
        @test Array(vector.continued.outputs.squared) == [0.015625, 0.0625]

        make(; kwargs...) = prepare_transpiled(PTE.vector_relaxation, [2.0, 4.0];
            backend, method=:advance!, argument=zeros(2), outputs=(squared=:squared,), kwargs...)
        program = make()
        other = make()
        initial = initial_transpiled_state(program)
        @test_throws ArgumentError other(initial, zeros(2))
        @test_throws ArgumentError program(initial, zeros(3))
        @test_throws ArgumentError program(initial, zeros(Float32, 2))
        @test_throws ArgumentError make(iterations=-1)
        for kwargs in ((iterations=0,), (kernel_kwargs=(rounds=0,),))
            idle = make(; kwargs...)
            @test Array(idle(initial_transpiled_state(idle), zeros(2)).outputs.squared) == [4.0, 16.0]
        end
        @test_throws ArgumentError prepare_transpiled(PTE.accumulating_value, 1.0;
            backend, method=:advance!, argument=2.0, outputs=(missing=:missing,))
        @test_throws ArgumentError prepare_transpiled(PTE.accumulating_value, 1.0;
            backend, method=:advance!, argument=2.0, outputs=(missing=(:missing,:value),))
        backend === :native && @test (@inferred program(initial, zeros(2))).outputs.squared == [0.25, 1.0]
    end
end

check_transpiled_program(:native)
