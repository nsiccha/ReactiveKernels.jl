module FiniteStaticIndexReactantTests

using ReactiveKernels, Reactant, Test
const RK = ReactiveKernels

struct FixedWrite{P,I}
    contract::P
end
function (operation::FixedWrite{P,I})(raw, value, active) where {P,I}
    RK._sm_finite_structural_write(operation.contract, raw, I, value, active)
end

@testset "finite writes trace fixed indices without host packed masks" begin
    prototype = [(scalar=Float64(i), vector=Float64[i, i + 1]) for i in 1:2]
    contract = RK._sm_finite_structural_contract(prototype)
    host = RK._sm_finite_structural_pack(contract, prototype)
    raw = Reactant.to_rarray(host)
    value = Reactant.to_rarray((scalar=9.0, vector=[7.0, 8.0]); track_numbers=true)
    for index in (1, 2, 0, 3)
        operation = FixedWrite{typeof(contract),index}(contract)
        active = Reactant.to_rarray(true; track_numbers=true)
        compiled = Reactant.compile(operation, (raw, value, active);
                                    sync=true, donated_args=:none)
        result = compiled(raw, value, active)
        @test Bool(result.overflow) == !(index in 1:2)
        for slot in 1:2
            decoded = RK._sm_finite_structural_read(contract, result.storage, slot).value
            @test Float64(decoded.scalar) == (slot == index ? 9 : slot)
            @test Array(decoded.vector) == (slot == index ? [7, 8] : [slot, slot + 1])
        end
        for (before, after) in zip(values(host), values(raw))
            @test Array(after) == before
        end
        inactive = compiled(raw, value, Reactant.to_rarray(false; track_numbers=true))
        @test !Bool(inactive.overflow)
        for (before, after) in zip(values(host), values(inactive.storage))
            @test Array(after) == before
        end
    end
end

end
