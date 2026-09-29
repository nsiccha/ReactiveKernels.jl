using Test, ReactiveKernels, Reactant

@kernel compiled_slice_passthrough_source(position, shared) = begin
    result = position.scale * sum(shared)
end

@testset "recipe-free structured passthrough compiler parity" begin
    if pkgversion(Reactant) == v"0.2.284"
        # Isolated without RK in repro_reactant_passthrough_loop.jl: MLIR's
        # greedy rewrite does not finish within the bounded compile budget.
        @test_skip false
    else
        batch = vectorize(compiled_slice_passthrough_source;
            have=(:position, :shared), want=(:position, :shared), batched=:position)
        @test isempty(plan(batch).recipes)
        position = (; scale=[1.0, 2.0, 3.0], curve=reshape(collect(1.0:6.0), 2, 3))
        shared = [2.0, 5.0]
        traced = map(Reactant.to_rarray, position)
        traced_shared = Reactant.to_rarray(shared)
        compiled = Reactant.@compile batch(traced, traced_shared)
        actual, repeated = compiled(traced, traced_shared)
        native, native_repeated = batch(position, shared)
        @test Array(actual.scale) == native.scale
        @test Array(actual.curve) == native.curve
        @test Array(repeated) == native_repeated
        @test Array(traced.curve) == position.curve
        @test Array(traced_shared) == shared
        later = map(Reactant.to_rarray, map(value -> value .+ 1, position))
        compiled(later, traced_shared)
        @test Array(actual.curve) == native.curve

        function operations(count)
            input = (; scale=Reactant.to_rarray(fill(2.0, count)),
                       curve=Reactant.to_rarray(fill(3.0, 2, count)))
            hlo = repr(Reactant.@code_hlo optimize=true batch(input, traced_shared))
            @test count_occurrences(hlo) == 1
            [m.match for m in eachmatch(r"(?:stablehlo|enzyme)\.[a-z_]+", hlo)]
        end
        count_occurrences(hlo) = count("stablehlo.while", hlo)
        small = operations(3)
        @test !isempty(small)
        @test small == operations(23)
    end
end
