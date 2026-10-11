using DifferentiationInterface, Distributions, Enzyme, ReactiveKernels, ReactiveKernelsPPL,
    Reactant, Test
using ReactiveKernels: compile_ad_value_and_gradient

# Compiled execution of the shared @plate cells of `test_plate_cell_once.jl`:
# a cell whose several per-index values share work is one RK plate returning
# a tuple, and its projections read that plate's lanes. Arrays per index are
# native only; these are the scalar and level shapes. Shared row outputs over
# traced slices compile in `test_plate_cells_reactant.jl` and
# `test_array_cell_rows_reactant.jl`; the `row_output` fixture here builds its
# row with an ordinary `Vector` literal, which compiled plate cells do not
# batch whether or not the cell is shared.

function _pco_compiled_ops(kernel, ru)
    hlo = repr(Reactant.@code_hlo optimize = false kernel(ru))
    ops = Dict{String,Int}()
    for m in eachmatch(r"(?:stablehlo|enzyme)\.[a-z_]+", hlo)
        ops[m.match] = get(ops, m.match, 0) + 1
    end
    ops
end

@testset "Reactant: a shared @plate cell's projections read its lanes" begin
    for kind in (:scalar_two_outputs, :cell_observations, :output_and_observation,
            :level_outputs)
        inventories = map((4, 11)) do n
            fx = _pco_fixture(kind, n)
            sampler = prepare_sampler(fx.built, fx.bound, fx.u;
                backend = AutoEnzyme(; mode = Enzyme.Reverse))
            kernel = prepare_query(fx.built, fx.bound, :sampler)
            ru = Reactant.to_rarray(fx.u)
            primal = Reactant.@compile kernel(ru)
            reverse = compile_ad_value_and_gradient(sampler.ad, ru)
            for u in (fx.u, fx.u .+ 0.11)
                input = Reactant.to_rarray(u)
                value, gradient = sampler_value_and_gradient!(sampler, similar(u), u)
                @test Float64(primal(input)) ≈ fx.oracle(u) rtol = 1e-9
                cv, cg = reverse(input)
                @test Float64(cv) ≈ value rtol = 1e-9
                @test Array(cg) ≈ gradient rtol = 1e-8 atol = 1e-9
            end
            _pco_compiled_ops(kernel, ru)
        end
        # The emitted program does not grow with the index count.
        @test inventories[1] == inventories[2]
    end
end
