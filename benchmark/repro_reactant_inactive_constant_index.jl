# Backend-only: a live branch traces an invalid read of a host constant
# even when every runtime lane takes the other arm. No RK imports.
using Reactant, Test
Reactant.set_default_backend("cpu")
const HOST_VALUE = [0.1]

function cell(x0)
    x = Reactant.@allowscalar x0[]
    y = zero(x)
    Reactant.@trace if x > 0
        y = 2x
    else
        y = HOST_VALUE[100]
    end
    return y
end
loss(v) = sum(only(Reactant.Ops.batch(cell, [v], Int64[length(v)])))

for n in (3, 5)
    v = fill(0.2, n)
    @test sum(x > 0 ? 2x : HOST_VALUE[100] for x in v) ≈ 0.4n
    rv = Reactant.to_rarray(v)
    err = try
        compiled = Reactant.@compile loss(rv)
        @test Float64(compiled(rv)) ≈ 0.4n
        nothing
    catch e
        e
    end
    if err !== nothing
        @test err isa BoundsError
        @test err.i == (100,)
        println("PINNED_BACKEND_LIMITATION ", sprint(showerror, err))
    else
        println("BACKEND_LIMITATION_LIFTED")
    end
end
